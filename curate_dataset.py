#!/usr/bin/env python3
"""
Diversity-Ordered Training Data Curation

Reorders a JSONL training dataset so that each training sequence (formed by
packing documents into fixed-length windows) contains maximum topic diversity.

Approach:
  1. Embed each document using the base model's own token embedding layer
     (mean-pooled, L2-normalized — no external embedding model needed)
  2. Cluster embeddings with MiniBatchKMeans (k=30 default)
  3. Reorder using a stratified greedy algorithm that maximizes cluster
     diversity within each training sequence

Supports incremental operation: if you append new samples to the input file
and re-run, only the new samples are embedded on GPU. Existing embeddings
and token counts are loaded from cache.

Requirements:
    pip install numpy torch transformers safetensors scikit-learn matplotlib
    pip install umap-learn  # optional, for UMAP cluster cloud visualization

Usage:
    # First run (embeds all samples on GPU):
    python curate_dataset.py \\
        --input training_data.jsonl \\
        --output training_data_curated.jsonl \\
        --model-dir /path/to/base-model

    # After appending new samples (only embeds the delta):
    python curate_dataset.py \\
        --input training_data.jsonl \\
        --output training_data_curated.jsonl \\
        --model-dir /path/to/base-model

    # Stats only (no output written):
    python curate_dataset.py \\
        --input training_data.jsonl \\
        --output training_data_curated.jsonl \\
        --model-dir /path/to/base-model \\
        --stats-only

    # Find optimal k for your dataset (sweeps k=5..100):
    python curate_dataset.py \\
        --input training_data.jsonl \\
        --output training_data_curated.jsonl \\
        --model-dir /path/to/base-model \\
        --calibrate-k
"""

import argparse
import json
import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Fix OpenBLAS threading issue that causes segfault in sklearn on some systems
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")


# ── Base-Model Embedding ─────────────────────────────────────────────────────


def load_embedding_layer(model_dir: str, device: str = "cuda:0") -> tuple:
    """
    Load only the token embedding layer from a HuggingFace model stored in
    safetensors format. Does NOT load the full transformer — just the
    embedding table (~1 GB for a typical model).

    Returns (embed_weight, tokenizer).
    """
    from safetensors import safe_open
    from transformers import AutoTokenizer

    log.info(f"Loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        embed_key = "model.embed_tokens.weight"
        embed_file = index["weight_map"][embed_key]
        embed_path = os.path.join(model_dir, embed_file)
    else:
        # Single safetensors file
        embed_key = "model.embed_tokens.weight"
        embed_path = os.path.join(model_dir, "model.safetensors")

    log.info(f"Loading embedding layer from {os.path.basename(embed_path)}")
    with safe_open(embed_path, framework="pt", device=str(device)) as f:
        embed_weight = f.get_tensor(embed_key)

    log.info(f"  Shape: {embed_weight.shape} ({embed_weight.dtype})")
    return embed_weight, tokenizer


def embed_texts(
    texts: list[str],
    model_dir: str,
    device: str = "cuda:0",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Embed texts using the base model's token embedding layer.

    For each document: tokenize (no truncation) -> lookup embeddings ->
    mean-pool over tokens -> L2-normalize.

    Returns (N, D) float32 numpy array.
    """
    embed_weight, tokenizer = load_embedding_layer(model_dir, device)
    vocab_size, hidden_dim = embed_weight.shape

    log.info(f"Embedding {len(texts)} texts (dim={hidden_dim})...")
    t0 = time.time()
    all_embeddings = np.zeros((len(texts), hidden_dim), dtype=np.float32)

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))

        if start % (batch_size * 50) == 0:
            elapsed = time.time() - t0
            rate = start / elapsed if elapsed > 0 else 0
            log.info(f"  {start:>6}/{len(texts)} ({rate:.0f} samples/s)")

        for i, text in enumerate(texts[start:end]):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            token_ids_t = torch.tensor(token_ids, device=device, dtype=torch.long)
            token_ids_t = token_ids_t.clamp(0, vocab_size - 1)

            with torch.inference_mode():
                token_embeds = embed_weight[token_ids_t]
                mean_embed = token_embeds.mean(dim=0)
                mean_embed = torch.nn.functional.normalize(mean_embed, dim=0)

            all_embeddings[start + i] = mean_embed.float().cpu().numpy()

    elapsed = time.time() - t0
    log.info(f"  Done in {elapsed:.1f}s ({len(texts) / elapsed:.0f} samples/s)")
    return all_embeddings


def load_or_embed(
    texts: list[str],
    emb_path: str,
    model_dir: str,
    device: str = "cuda:0",
    batch_size: int = 64,
    force_load: str | None = None,
) -> np.ndarray:
    """
    Load cached embeddings, or embed incrementally if new samples were appended.
    """
    n_total = len(texts)

    if force_load:
        log.info(f"Loading precomputed embeddings from {force_load}")
        embeddings = np.load(force_load, allow_pickle=False)
        assert len(embeddings) == n_total, (
            f"Embedding count {len(embeddings)} != text count {n_total}. "
            f"Remove --load-embeddings to trigger incremental embedding."
        )
        nan_mask = np.any(np.isnan(embeddings), axis=1)
        n_nan = int(nan_mask.sum())
        if n_nan > 0:
            log.warning(
                f"  {n_nan} embeddings are NaN (likely OOM-skipped). "
                f"Replacing with zero vectors — these will form a separate cluster."
            )
            embeddings[nan_mask] = 0
        return embeddings

    if os.path.exists(emb_path):
        cached = np.load(emb_path, allow_pickle=False)
        n_cached = len(cached)
        log.info(f"Found cached embeddings: {cached.shape}")

        if n_cached == n_total:
            log.info("  Cache is up-to-date, skipping GPU.")
            return cached
        elif n_cached < n_total:
            n_new = n_total - n_cached
            log.info(f"  Embedding {n_new} new samples incrementally...")
            new_embeddings = embed_texts(
                texts[n_cached:], model_dir, device, batch_size
            )
            embeddings = np.concatenate([cached, new_embeddings], axis=0)
            np.save(emb_path, embeddings)
            log.info(f"  Updated cache: {embeddings.shape}")
            return embeddings
        else:
            log.warning(
                f"  Cache has MORE rows ({n_cached}) than dataset ({n_total}). Re-embedding."
            )

    log.info("No valid cache found, embedding from scratch...")
    embeddings = embed_texts(texts, model_dir, device, batch_size)
    np.save(emb_path, embeddings)
    log.info(f"  Saved to {emb_path}")
    return embeddings


# ── Token Counting ────────────────────────────────────────────────────────────


def load_or_count_tokens(
    texts: list[str],
    tok_path: str,
    model_dir: str,
) -> np.ndarray:
    """
    Load cached token counts or compute them. Supports incremental updates.
    """
    from transformers import AutoTokenizer

    n_total = len(texts)

    if os.path.exists(tok_path):
        cached = np.load(tok_path, allow_pickle=False)
        if len(cached) == n_total:
            log.info(f"Token counts loaded from cache: {len(cached)} samples")
            return cached
        elif len(cached) < n_total:
            n_new = n_total - len(cached)
            log.info(f"Counting {n_new} new samples incrementally...")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            new_counts = np.array(
                [
                    len(tokenizer.encode(t, add_special_tokens=False))
                    for t in texts[len(cached) :]
                ],
                dtype=np.int64,
            )
            counts = np.concatenate([cached, new_counts])
            np.save(tok_path, counts)
            return counts
        else:
            log.warning("Token cache outdated. Recomputing.")

    log.info(f"Computing token counts for {n_total} samples...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    counts = np.array(
        [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts],
        dtype=np.int64,
    )
    log.info(f"  Done in {time.time() - t0:.1f}s")
    np.save(tok_path, counts)
    return counts


# ── K Calibration ─────────────────────────────────────────────────────────────


def _gpu_cosine_distances(embeddings: np.ndarray, device: str = "cuda:0") -> np.ndarray:
    """Compute pairwise cosine distance matrix on GPU, return as CPU numpy.

    For L2-normalized embeddings: cosine_distance = 1 - X @ X.T
    Processes in chunks to avoid GPU OOM on large datasets.
    """
    n = len(embeddings)
    log.info(f"  Computing {n:,}×{n:,} cosine distance matrix on GPU...")
    t0 = time.time()

    X = torch.from_numpy(embeddings).to(device)
    # Ensure L2-normalized
    X = X / (torch.linalg.norm(X, dim=1, keepdim=True) + 1e-8)

    # Compute in chunks to limit GPU memory (~4 GB per chunk)
    chunk_size = max(1, min(8192, int(4e9 / (n * 4))))
    dist = np.empty((n, n), dtype=np.float32)

    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        sim_chunk = (X[i:end] @ X.T).cpu().numpy()
        dist[i:end] = np.maximum(0.0, 1.0 - sim_chunk)  # clamp fp rounding

    del X
    torch.cuda.empty_cache()
    log.info(f"  Distance matrix computed in {time.time() - t0:.1f}s")
    return dist


def calibrate_k(
    embeddings: np.ndarray,
    k_values: list[int] | None = None,
    sample_size: int = 10000,
    seed: int = 42,
) -> list[dict]:
    """
    Sweep k values and report silhouette scores to help choose n_clusters.

    Uses a random subsample for silhouette (which is O(N^2)) while fitting
    KMeans on all data. Uses GPU for pairwise distance computation when
    sample_size > 20000 (avoids slow single-threaded CPU BLAS).
    Returns a list of {k, silhouette, sizes_min, sizes_median, sizes_max} dicts.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    if k_values is None:
        k_values = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]

    rng = np.random.RandomState(seed)
    n = len(embeddings)
    actual_sample = min(sample_size, n)
    sample_idx = rng.choice(n, actual_sample, replace=False)

    # Precompute distance matrix on GPU for large samples (much faster than
    # sklearn's CPU-based cosine distance with single-threaded netlib BLAS)
    use_precomputed = actual_sample > 20000 and torch.cuda.is_available()
    if use_precomputed:
        sil_input = _gpu_cosine_distances(embeddings[sample_idx])
        sil_metric = "precomputed"
    else:
        sil_metric = "cosine"
        sil_input = embeddings[sample_idx]

    results = []
    log.info(
        f"Calibrating k ({len(k_values)} values, silhouette on {actual_sample:,} samples, "
        f"metric={'GPU precomputed' if use_precomputed else 'cosine'})..."
    )

    for k in k_values:
        if k >= n:
            log.warning(f"  k={k} >= N={n}, skipping")
            continue
        t0 = time.time()
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=3)
        labels = km.fit_predict(embeddings)

        sil = silhouette_score(sil_input, labels[sample_idx], metric=sil_metric)
        _, counts = np.unique(labels, return_counts=True)
        elapsed = time.time() - t0

        results.append(
            {
                "k": k,
                "silhouette": round(sil, 4),
                "sizes_min": int(counts.min()),
                "sizes_median": int(np.median(counts)),
                "sizes_max": int(counts.max()),
            }
        )
        log.info(
            f"  k={k:>4d}: silhouette={sil:.4f}  "
            f"sizes=[{counts.min()}, {int(np.median(counts))}, {counts.max()}]  "
            f"({elapsed:.1f}s)"
        )

    # Print recommendation
    best = max(results, key=lambda r: r["silhouette"])
    log.info(f"\n  Peak silhouette: k={best['k']} ({best['silhouette']})")

    # Find plateau: highest k within 5% of peak silhouette
    threshold = best["silhouette"] * 0.95
    plateau = [r for r in results if r["silhouette"] >= threshold]
    recommended = max(plateau, key=lambda r: r["k"])
    if recommended["k"] != best["k"]:
        log.info(
            f"  Recommended: k={recommended['k']} (within 5% of peak, "
            f"more diversity headroom)"
        )
    else:
        log.info(f"  Recommended: k={best['k']}")

    return results


# ── Clustering ────────────────────────────────────────────────────────────────


def cluster(
    embeddings: np.ndarray,
    n_clusters: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster embeddings with MiniBatchKMeans. Returns (labels, distances)."""
    from sklearn.cluster import MiniBatchKMeans

    log.info(f"Clustering {len(embeddings)} embeddings into {n_clusters} clusters...")
    t0 = time.time()

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=4096,
        n_init=3,
    )
    labels = kmeans.fit_predict(embeddings)
    distances = np.linalg.norm(embeddings - kmeans.cluster_centers_[labels], axis=1)

    unique, counts = np.unique(labels, return_counts=True)
    log.info(
        f"  Done in {time.time() - t0:.1f}s. Sizes: min={counts.min()}, median={int(np.median(counts))}, max={counts.max()}"
    )
    return labels, distances


# ── Stratified Greedy Ordering ────────────────────────────────────────────────


def stratified_greedy_order(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Sample-count stratified greedy ordering.

    At each position, picks the sample from the most underrepresented cluster
    (largest deficit between target and actual proportion). This ensures every
    fixed-size window of the output contains near-maximum cluster diversity.
    """
    log.info("Building stratified greedy ordering...")
    t0 = time.time()

    N = len(labels)
    cluster_pools = defaultdict(list)
    for idx in range(N):
        cluster_pools[labels[idx]].append(idx)

    cluster_budget = np.zeros(n_clusters, dtype=np.int64)
    order = []
    pointers = {k: 0 for k in range(n_clusters)}
    total_remaining = N

    while len(order) < N:
        total_picked = len(order)
        best_cluster = None
        best_deficit = -float("inf")

        for k in range(n_clusters):
            pool = cluster_pools.get(k, [])
            if pointers[k] >= len(pool):
                continue
            remaining_k = len(pool) - pointers[k]
            target_share = remaining_k / max(total_remaining, 1)
            actual_share = cluster_budget[k] / max(total_picked, 1)
            deficit = target_share - actual_share
            if deficit > best_deficit:
                best_deficit = deficit
                best_cluster = k

        if best_cluster is None:
            break

        idx = cluster_pools[best_cluster][pointers[best_cluster]]
        pointers[best_cluster] += 1
        order.append(idx)
        cluster_budget[best_cluster] += 1
        total_remaining -= 1

    log.info(f"  Done in {time.time() - t0:.1f}s ({len(order)} samples)")
    return np.array(order)


# ── Diversity Measurement ─────────────────────────────────────────────────────


def compute_window_diversities(
    order: np.ndarray,
    labels: np.ndarray,
    token_counts: np.ndarray,
    seq_len: int,
) -> np.ndarray:
    """
    Compute per-window cluster diversity, simulating document packing.

    Models a real data loader: documents are packed into fixed-length token
    windows. If a document overflows the current window, the window is flushed
    and the document starts a new one (documents that exceed seq_len span
    multiple windows).
    """
    window_clusters = set()
    window_tokens = 0
    diversities = []

    for idx in order:
        doc_tokens = int(token_counts[idx])
        doc_label = labels[idx]

        if window_tokens + doc_tokens <= seq_len:
            # Fits in current window
            window_clusters.add(doc_label)
            window_tokens += doc_tokens
        else:
            # Flush current window (if non-empty)
            if window_tokens > 0:
                diversities.append(len(window_clusters))
            # Start new window(s) with this document
            window_clusters = {doc_label}
            window_tokens = doc_tokens
            # Handle documents longer than seq_len
            while window_tokens >= seq_len:
                diversities.append(len(window_clusters))
                window_tokens -= seq_len
                if window_tokens > 0:
                    window_clusters = {doc_label}
                else:
                    window_clusters = set()

    if window_clusters:
        diversities.append(len(window_clusters))

    return np.array(diversities) if diversities else np.array([0])


def measure_token_window_diversity(
    order: np.ndarray,
    labels: np.ndarray,
    token_counts: np.ndarray,
    seq_len: int,
) -> tuple[dict, np.ndarray]:
    """
    Measure cluster diversity per training sequence (token-budget windows).
    Returns (summary_dict, raw_diversities_array).
    """
    divs = compute_window_diversities(order, labels, token_counts, seq_len)
    stats = {
        "mean": round(float(divs.mean()), 2),
        "min": int(divs.min()),
        "max": int(divs.max()),
        "std": round(float(divs.std()), 2),
        "n_sequences": len(divs),
    }
    return stats, divs


def _draw_cluster_bar(
    ax,
    order: np.ndarray,
    labels: np.ndarray,
    cmap,
    n_clusters: int,
    title: str,
):
    """Draw a thin color-bar showing cluster assignment across dataset order."""
    colors = cmap(labels[order] / max(n_clusters - 1, 1))
    # Draw as a 1-pixel-tall image stretched vertically
    ax.imshow(
        colors[np.newaxis, :, :3],
        aspect="auto",
        interpolation="none",
        extent=[0, len(order), 0, 1],
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim(0, len(order))
    # Show sample indices on x-axis
    n = len(order)
    tick_positions = np.linspace(0, n, min(8, n), dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{t:,}" for t in tick_positions], fontsize=7)


def plot_ordering_comparison(
    original_order: np.ndarray,
    curated_order: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    seq_len: int,
    orig_div: dict,
    curated_div: dict,
    orig_divs: np.ndarray,
    curated_divs: np.ndarray,
    save_path: str,
):
    """
    Generate a comparison chart: original vs curated ordering.

    Top two panels show color-coded cluster assignment per sample position.
    Bottom panel shows per-window diversity line chart.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = matplotlib.colormaps["tab20"].resampled(n_clusters)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 6),
        gridspec_kw={"height_ratios": [1, 1, 2.5]},
    )
    seq_label = f"{seq_len // 1024}K" if seq_len >= 1024 else str(seq_len)
    fig.suptitle(
        f"Ordering Comparison ({n_clusters} clusters, {seq_label}-token windows)",
        fontsize=12,
        fontweight="bold",
    )

    # Original order bar
    _draw_cluster_bar(
        axes[0],
        original_order,
        labels,
        cmap,
        n_clusters,
        f"Original Order (window diversity: mean={orig_div['mean']}, "
        f"min={orig_div['min']}, std={orig_div['std']})",
    )

    # Curated order bar
    _draw_cluster_bar(
        axes[1],
        curated_order,
        labels,
        cmap,
        n_clusters,
        f"Curated Order (window diversity: mean={curated_div['mean']}, "
        f"min={curated_div['min']}, std={curated_div['std']})",
    )

    # Diversity line chart
    ax = axes[2]

    ax.fill_between(
        range(len(orig_divs)),
        orig_divs,
        alpha=0.3,
        color="tab:red",
    )
    ax.plot(
        orig_divs,
        color="tab:red",
        linewidth=0.8,
        alpha=0.7,
        label=f"Original (mean={orig_div['mean']}, min={orig_div['min']})",
    )
    ax.fill_between(
        range(len(curated_divs)),
        curated_divs,
        alpha=0.3,
        color="tab:blue",
    )
    ax.plot(
        curated_divs,
        color="tab:blue",
        linewidth=0.8,
        alpha=0.7,
        label=f"Curated (mean={curated_div['mean']}, min={curated_div['min']})",
    )
    ax.axhline(
        n_clusters,
        color="gray",
        linestyle="--",
        linewidth=0.5,
        label=f"Maximum ({n_clusters})",
    )

    ax.set_xlabel("Training sequence index", fontsize=9)
    ax.set_ylabel("Unique clusters\nper window", fontsize=9)
    ax.set_title(
        f"Window Diversity Over Dataset Position ({seq_label}-token windows)",
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(0, n_clusters + 2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved ordering comparison: {save_path}")


def plot_cluster_cloud(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    save_path: str,
    umap_cache_path: str | None = None,
):
    """
    UMAP cluster cloud with centroid labels.

    Projects embeddings to 2D via PCA(50) -> UMAP(2), then scatter-plots
    each point colored by cluster with centroid labels showing cluster ID
    and sample count. Falls back to a bar chart if umap-learn is not installed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import umap
    except ImportError:
        log.warning("  umap-learn not installed, generating bar chart instead")
        _plot_cluster_sizes_fallback(labels, n_clusters, save_path)
        return

    # Compute or load UMAP projection
    umap2d = None
    if umap_cache_path and os.path.exists(umap_cache_path):
        cached = np.load(umap_cache_path, allow_pickle=False)
        if len(cached) == len(embeddings):
            umap2d = cached
            log.info(f"  UMAP projection loaded from cache ({umap_cache_path})")

    if umap2d is None:
        from sklearn.decomposition import PCA

        log.info("  Computing UMAP projection (PCA 50 -> UMAP 2)...")
        t0 = time.time()
        pca = PCA(n_components=min(50, embeddings.shape[1]), random_state=42)
        emb_pca = pca.fit_transform(embeddings)
        reducer = umap.UMAP(
            n_components=2, random_state=42, n_neighbors=15, min_dist=0.1
        )
        umap2d = reducer.fit_transform(emb_pca)
        log.info(f"  UMAP done in {time.time() - t0:.1f}s")

        if umap_cache_path:
            np.save(umap_cache_path, umap2d)
            log.info(f"  Saved UMAP cache: {umap_cache_path}")

    # Plot
    cmap = matplotlib.colormaps["tab20"].resampled(n_clusters)
    unique, counts = np.unique(labels, return_counts=True)
    cluster_count = dict(zip(unique, counts))

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = cmap(labels / max(n_clusters - 1, 1))

    ax.scatter(
        umap2d[:, 0],
        umap2d[:, 1],
        c=colors[:, :3],
        s=1,
        alpha=0.3,
        rasterized=True,
    )

    # Centroid labels
    for k in unique:
        mask = labels == k
        cx = umap2d[mask, 0].mean()
        cy = umap2d[mask, 1].mean()
        ax.annotate(
            f"c{k} ({cluster_count[k]:,})",
            (cx, cy),
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.8,
                edgecolor="gray",
                linewidth=0.5,
            ),
        )

    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title(
        f"UMAP Cluster Cloud ({len(labels):,} samples, {n_clusters} clusters)",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved cluster cloud: {save_path}")


def _plot_cluster_sizes_fallback(
    labels: np.ndarray,
    n_clusters: int,
    save_path: str,
):
    """Bar chart fallback when umap-learn is not installed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique, counts = np.unique(labels, return_counts=True)
    sorted_idx = np.argsort(-counts)

    cmap = matplotlib.colormaps["tab20"].resampled(n_clusters)
    colors = [cmap(unique[i] / max(n_clusters - 1, 1)) for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(counts)), counts[sorted_idx], color=colors, edgecolor="none")
    ax.set_xlabel("Cluster (sorted by size)", fontsize=9)
    ax.set_ylabel("Number of samples", fontsize=9)
    ax.set_title(
        f"Cluster Size Distribution ({n_clusters} clusters, {len(labels):,} samples)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(
        [str(int(unique[i])) for i in sorted_idx],
        fontsize=6,
        rotation=45,
    )
    ax.axhline(
        np.median(counts),
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label=f"median={int(np.median(counts))}",
    )
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved cluster sizes (fallback): {save_path}")


# ── Cosine Similarity Analysis ────────────────────────────────────────────────


def analyze_similarity(
    embeddings: np.ndarray,
    sample_size: int = 5000,
    seed: int = 42,
) -> dict:
    """Pairwise cosine similarity stats on a random sample."""
    log.info(f"Analyzing cosine similarity (sample={sample_size})...")
    rng = np.random.RandomState(seed)
    n = len(embeddings)
    sample = embeddings[rng.choice(n, min(sample_size, n), replace=False)]

    sim_matrix = sample @ sample.T
    np.fill_diagonal(sim_matrix, 0)
    sims = sim_matrix[np.triu_indices(len(sample), k=1)]

    return {
        "mean": round(float(np.mean(sims)), 4),
        "std": round(float(np.std(sims)), 4),
        "median": round(float(np.median(sims)), 4),
        "high_sim_pairs_gt_0.9": int(np.sum(sims > 0.9)),
        "total_pairs_sampled": int(len(sims)),
    }


# ── Main Pipeline ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Diversity-ordered training data curation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="HuggingFace model directory (for embedding layer + tokenizer)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=30,
        help="Number of clusters for diversity ordering",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=131072,
        help="Training sequence length in tokens (for diversity measurement)",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="JSON field containing the document text",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only compute stats, don't write output",
    )
    parser.add_argument(
        "--calibrate-k",
        action="store_true",
        help="Sweep k values and report silhouette scores to help choose --n-clusters, then exit",
    )
    parser.add_argument(
        "--calibrate-k-samples",
        type=int,
        default=10000,
        help="Number of samples for silhouette computation during k-sweep. "
        "0 = use all samples (O(N^2), slow for large datasets). Default: 10000",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating visualization PNGs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--load-embeddings",
        type=str,
        default=None,
        help="Force-load embeddings from .npy (skip GPU)",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=0,
        help="Apply PCA to reduce embedding dimensionality before clustering. "
        "0 = disabled (default). Recommended: 50-100 when D > N (e.g., "
        "2880-dim deep embeddings with <2000 samples).",
    )
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    base, _ = os.path.splitext(args.output)
    emb_path = f"{base}_embeddings.npy"
    input_base, _ = os.path.splitext(args.input)
    tok_path = f"{input_base}_token_counts.npy"

    # ── Validate inputs ──
    if not os.path.isdir(args.model_dir):
        parser.error(f"--model-dir does not exist: {args.model_dir}")

    # ── Load dataset ──
    log.info(f"Loading {args.input}")
    lines = []
    texts = []
    with open(args.input) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON at line {lineno}: {e}") from e
            text = obj.get(args.text_field)
            if text is None:
                raise KeyError(
                    f"Field '{args.text_field}' not found at line {lineno}. "
                    f"Available: {list(obj.keys())}. Use --text-field to specify."
                )
            lines.append(line)
            texts.append(text)

    N = len(texts)
    if N == 0:
        log.error("Input file is empty or contains no valid records.")
        return
    log.info(f"  {N:,} examples ({sum(len(t) for t in texts) / 1e6:.1f}M chars)")

    # ── Embed ──
    embeddings = load_or_embed(
        texts,
        emb_path,
        args.model_dir,
        device,
        args.embed_batch_size,
        args.load_embeddings,
    )

    # ── Calibrate k (optional) ──
    if args.calibrate_k:
        sample_size = (
            args.calibrate_k_samples
            if args.calibrate_k_samples > 0
            else len(embeddings)
        )
        calibrate_k(embeddings, sample_size=sample_size, seed=args.seed)
        return

    # ── Token counts ──
    token_counts = load_or_count_tokens(texts, tok_path, args.model_dir)
    del texts  # Free text strings — embeddings and token counts are cached
    total_tokens = int(token_counts.sum())
    log.info(
        f"  Total tokens: {total_tokens:,} (~{total_tokens // args.seq_len} training sequences)"
    )

    # ── PCA (optional) ──
    pca_info = None
    if args.pca_components > 0:
        from sklearn.decomposition import PCA

        d_orig = embeddings.shape[1]
        n_comp = min(args.pca_components, embeddings.shape[0] - 1, d_orig)
        log.info(f"Applying PCA: {d_orig} -> {n_comp} components...")
        t0 = time.time()
        pca = PCA(n_components=n_comp, random_state=args.seed)
        embeddings = pca.fit_transform(embeddings)
        explained = pca.explained_variance_ratio_.sum()
        # Re-normalize after PCA
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)
        log.info(
            f"  PCA done in {time.time() - t0:.1f}s. "
            f"Explained variance: {explained:.1%} ({n_comp} components)"
        )
        pca_info = {
            "n_components": n_comp,
            "explained_variance": round(float(explained), 4),
            "original_dim": d_orig,
        }

    # ── Similarity ──
    sim_stats = analyze_similarity(embeddings, seed=args.seed)

    # ── Cluster ──
    labels, distances = cluster(embeddings, args.n_clusters, args.seed)

    # ── Order ──
    order = stratified_greedy_order(labels, args.n_clusters)

    # ── Measure ──
    orig_div, orig_divs = measure_token_window_diversity(
        np.arange(N), labels, token_counts, args.seq_len
    )
    curated_div, curated_divs = measure_token_window_diversity(
        order, labels, token_counts, args.seq_len
    )

    log.info(f"  Original:  {orig_div}")
    log.info(f"  Curated:   {curated_div}")
    log.info(
        f"  Improvement: mean {orig_div['mean']} -> {curated_div['mean']}, "
        f"min {orig_div['min']} -> {curated_div['min']}"
    )

    # ── Write ──
    if not args.stats_only:
        log.info(f"Writing {args.output}")
        with open(args.output, "w") as f:
            for idx in order:
                f.write(lines[idx])
        log.info(
            f"  {len(order):,} examples ({os.path.getsize(args.output) / 1e6:.0f} MB)"
        )

    # ── Plot ──
    if not args.no_plot:
        plot_dir = os.path.dirname(args.output) or "."
        plot_ordering_comparison(
            original_order=np.arange(N),
            curated_order=order,
            labels=labels,
            n_clusters=args.n_clusters,
            seq_len=args.seq_len,
            orig_div=orig_div,
            curated_div=curated_div,
            orig_divs=orig_divs,
            curated_divs=curated_divs,
            save_path=os.path.join(plot_dir, "ordering_comparison.png"),
        )
        plot_cluster_cloud(
            embeddings=embeddings,
            labels=labels,
            n_clusters=args.n_clusters,
            save_path=os.path.join(plot_dir, "cluster_cloud.png"),
            umap_cache_path=f"{base}_umap2d.npy",
        )

    # ── Metadata ──
    unique, counts = np.unique(labels, return_counts=True)
    meta = {
        "input": args.input,
        "output": args.output,
        "input_count": N,
        "final_count": len(order),
        "n_clusters": args.n_clusters,
        "ordering": "stratified_greedy",
        "sequence_length": args.seq_len,
        "total_tokens": total_tokens,
        "n_training_sequences": curated_div["n_sequences"],
        "embedding_method": (
            "external_precomputed"
            if args.load_embeddings
            else "base_model_embed_tokens_mean_pool"
        ),
        "load_embeddings_path": args.load_embeddings,
        "pca": pca_info,
        "token_window_diversity": {"original": orig_div, "curated": curated_div},
        "similarity_stats": sim_stats,
        "cluster_sizes": {
            str(int(u)): int(c)
            for u, c in sorted(zip(unique, counts), key=lambda x: -x[1])
        },
    }
    meta_path = f"{base}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
