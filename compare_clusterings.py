#!/usr/bin/env python3
"""
Compare clustering quality between shallow (token embedding) and deep
(last-layer hidden state) embeddings.

Loads two .npy embedding files, clusters both with k-means, and reports:
  - Adjusted Rand Index (ARI): How similar are the cluster assignments?
  - Normalized Mutual Information (NMI): How much information do they share?
  - Per-sample cosine similarity between the two embedding spaces
  - Side-by-side UMAP visualization (if umap-learn is installed)

Interpretation:
  ARI > 0.7: Deep embeddings are largely redundant — same clusters.
  ARI 0.3-0.7: Moderate overlap — deep captures some new structure.
  ARI < 0.3: Very different clusterings — deep embeddings are worth pursuing.

Usage:
    python compare_clusterings.py \
        --shallow shallow_embeddings.npy \
        --deep deep_embeddings.npy \
        --n-clusters 30 \
        --pca-components 100
"""

import argparse
import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int,
    pca_components: int = 0,
    seed: int = 42,
    label: str = "",
) -> np.ndarray:
    """Cluster embeddings with optional PCA, returns labels."""
    from sklearn.cluster import MiniBatchKMeans

    emb = embeddings.copy()

    if pca_components > 0:
        from sklearn.decomposition import PCA

        n_comp = min(pca_components, emb.shape[0] - 1, emb.shape[1])
        pca = PCA(n_components=n_comp, random_state=seed)
        emb = pca.fit_transform(emb)
        explained = pca.explained_variance_ratio_.sum()
        # Re-normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-8)
        log.info(
            f"  [{label}] PCA {embeddings.shape[1]} -> {n_comp} ({explained:.1%} variance)"
        )

    km = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed, batch_size=4096, n_init=3
    )
    labels = km.fit_predict(emb)

    unique, counts = np.unique(labels, return_counts=True)
    log.info(
        f"  [{label}] Clustered into {len(unique)} groups. "
        f"Sizes: min={counts.min()}, median={int(np.median(counts))}, max={counts.max()}"
    )
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Compare shallow vs deep embedding clusterings."
    )
    parser.add_argument("--shallow", required=True, help="Shallow embeddings .npy file")
    parser.add_argument("--deep", required=True, help="Deep embeddings .npy file")
    parser.add_argument("--n-clusters", type=int, default=30)
    parser.add_argument(
        "--pca-components",
        type=int,
        default=0,
        help="Apply PCA before clustering (0 = disabled). "
        "Applied to BOTH embedding sets for fair comparison.",
    )
    parser.add_argument(
        "--pca-deep-only",
        type=int,
        default=0,
        help="Apply PCA only to deep embeddings (0 = disabled). "
        "Use when deep has D > N but shallow doesn't need it.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Save side-by-side UMAP comparison to this path",
    )
    args = parser.parse_args()

    # Load
    log.info(f"Loading shallow: {args.shallow}")
    shallow = np.load(args.shallow, allow_pickle=False)
    log.info(f"  Shape: {shallow.shape}")

    log.info(f"Loading deep: {args.deep}")
    deep = np.load(args.deep, allow_pickle=False)
    log.info(f"  Shape: {deep.shape}")

    assert len(shallow) == len(deep), (
        f"Mismatch: shallow has {len(shallow)} samples, deep has {len(deep)}"
    )
    n = len(shallow)

    # Cluster
    log.info(f"\nClustering with k={args.n_clusters}...")

    shallow_pca = args.pca_components
    deep_pca = args.pca_deep_only if args.pca_deep_only > 0 else args.pca_components

    shallow_labels = cluster_embeddings(
        shallow,
        args.n_clusters,
        pca_components=shallow_pca,
        seed=args.seed,
        label="shallow",
    )
    deep_labels = cluster_embeddings(
        deep, args.n_clusters, pca_components=deep_pca, seed=args.seed, label="deep"
    )

    # Compare
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    ari = adjusted_rand_score(shallow_labels, deep_labels)
    nmi = normalized_mutual_info_score(shallow_labels, deep_labels)

    log.info(f"\n{'=' * 60}")
    log.info(f"COMPARISON RESULTS")
    log.info(f"{'=' * 60}")
    log.info(f"  Adjusted Rand Index (ARI): {ari:.4f}")
    log.info(f"  Normalized Mutual Info:    {nmi:.4f}")

    if ari > 0.7:
        log.info(f"  -> HIGH overlap: Deep embeddings produce similar clusters.")
        log.info(f"     Deep embeddings may not add value for ordering.")
    elif ari > 0.3:
        log.info(f"  -> MODERATE overlap: Deep captures some new structure.")
        log.info(f"     Worth testing in a training ablation.")
    else:
        log.info(f"  -> LOW overlap: Very different clusterings!")
        log.info(f"     Deep embeddings capture meaningfully different structure.")

    # Cluster size summary
    from collections import Counter

    shallow_sizes = Counter(shallow_labels)
    deep_sizes = Counter(deep_labels)
    log.info(f"\nCluster counts:")
    log.info(f"  Shallow: {len(shallow_sizes)}")
    log.info(f"  Deep:    {len(deep_sizes)}")

    # Per-sample embedding similarity (only meaningful if same dimensionality)
    if shallow.shape[1] == deep.shape[1]:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        s_t = torch.from_numpy(shallow).to(device)
        d_t = torch.from_numpy(deep).to(device)
        s_t = torch.nn.functional.normalize(s_t, dim=1)
        d_t = torch.nn.functional.normalize(d_t, dim=1)
        per_sample_sim = (s_t * d_t).sum(dim=1).cpu().numpy()
        del s_t, d_t

        log.info(f"\nPer-sample cosine similarity (shallow vs deep):")
        log.info(
            f"  mean={per_sample_sim.mean():.4f}, "
            f"std={per_sample_sim.std():.4f}, "
            f"min={per_sample_sim.min():.4f}, "
            f"max={per_sample_sim.max():.4f}"
        )
    else:
        log.info(
            f"\nDimensions differ ({shallow.shape[1]} vs {deep.shape[1]}), "
            f"skipping per-sample cosine similarity."
        )

    # UMAP visualization
    if args.plot:
        _plot_comparison(
            shallow,
            deep,
            shallow_labels,
            deep_labels,
            args.n_clusters,
            ari,
            nmi,
            args.plot,
            args.seed,
            shallow_pca,
            deep_pca,
        )

    log.info(f"\nDone.")


def _plot_comparison(
    shallow,
    deep,
    shallow_labels,
    deep_labels,
    n_clusters,
    ari,
    nmi,
    save_path,
    seed,
    shallow_pca,
    deep_pca,
):
    """Side-by-side UMAP scatter comparing the two clusterings."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import umap
    except ImportError:
        log.warning("umap-learn not installed, skipping visualization.")
        return

    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        f"Shallow vs Deep Embedding Clusters (k={n_clusters}, ARI={ari:.3f}, NMI={nmi:.3f})",
        fontsize=13,
        fontweight="bold",
    )

    cmap = matplotlib.colormaps["tab20"].resampled(n_clusters)

    for ax, emb, labels, title in [
        (axes[0], shallow, shallow_labels, "Shallow (token embedding)"),
        (axes[1], deep, deep_labels, "Deep (last-layer hidden state)"),
    ]:
        # PCA -> UMAP
        n_comp = min(50, emb.shape[1], emb.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=seed)
        emb_pca = pca.fit_transform(emb)
        reducer = umap.UMAP(
            n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1
        )
        umap2d = reducer.fit_transform(emb_pca)

        colors = cmap(labels / max(n_clusters - 1, 1))
        ax.scatter(umap2d[:, 0], umap2d[:, 1], c=colors[:, :3], s=8, alpha=0.6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        # Annotate cluster centroids
        for k in np.unique(labels):
            mask = labels == k
            cx, cy = umap2d[mask, 0].mean(), umap2d[mask, 1].mean()
            ax.annotate(
                str(k),
                (cx, cy),
                fontsize=6,
                fontweight="bold",
                ha="center",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    alpha=0.7,
                    linewidth=0.3,
                ),
            )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved comparison plot: {save_path}")


if __name__ == "__main__":
    main()
