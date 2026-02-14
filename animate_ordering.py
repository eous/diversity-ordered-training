#!/usr/bin/env python3
"""
Animated visualization of diversity-ordered training data curation.

Generates a video showing the stratified greedy ordering being built:
- Single color bar: curated order (bright, left) ← | → original order (dimmed, right)
- Items visually move from their original position to their curated position
- Bottom: diversity line chart building in real-time

Usage:
    python animate_ordering.py \
        --embeddings embeddings.npy \
        --token-counts token_counts.npy \
        --n-clusters 30 --seq-len 131072
"""

import argparse
import json
import logging
import os
import tempfile
import time

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def cluster_and_order(embeddings, n_clusters, seed=42):
    """Cluster embeddings and compute stratified greedy order."""
    from sklearn.cluster import MiniBatchKMeans
    from collections import defaultdict

    km = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed, batch_size=4096, n_init=3
    )
    labels = km.fit_predict(embeddings)

    # Stratified greedy ordering
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

    return labels, np.array(order)


def compute_window_info(order, labels, token_counts, seq_len):
    """
    Pre-compute per-sample cumulative state for animation.
    Returns:
        window_ids: array of which window each sample (in order) belongs to
        window_diversities: diversity count when each window completes
        window_boundaries: sample indices where windows complete
    """
    window_clusters = set()
    window_tokens = 0
    current_window = 0

    window_ids = np.zeros(len(order), dtype=np.int32)
    window_diversities = []
    window_boundaries = []

    for i, idx in enumerate(order):
        doc_tokens = int(token_counts[idx])
        doc_label = labels[idx]

        if window_tokens + doc_tokens <= seq_len:
            window_clusters.add(doc_label)
            window_tokens += doc_tokens
            window_ids[i] = current_window
        else:
            if window_tokens > 0:
                window_diversities.append(len(window_clusters))
                window_boundaries.append(i)
                current_window += 1
            window_clusters = {doc_label}
            window_tokens = doc_tokens
            window_ids[i] = current_window
            while window_tokens >= seq_len:
                window_diversities.append(len(window_clusters))
                window_boundaries.append(i)
                current_window += 1
                window_tokens -= seq_len
                if window_tokens > 0:
                    window_clusters = {doc_label}
                else:
                    window_clusters = set()

    if window_clusters:
        window_diversities.append(len(window_clusters))
        window_boundaries.append(len(order))

    return window_ids, np.array(window_diversities), np.array(window_boundaries)


def generate_frames(
    labels,
    order,
    token_counts,
    n_clusters,
    seq_len,
    frame_dir,
    n_frames=600,
    fig_width=14,
    fig_height=4.5,
):
    """Generate PNG frames for the animation.

    Single unified bar: left = curated order (bright, growing),
    right = remaining original order (dimmed, shrinking).
    Items visually migrate from the dimmed right side to the bright left side.
    """
    N = len(order)
    samples_per_frame = max(1, N // n_frames)

    cmap = matplotlib.colormaps["tab20"].resampled(n_clusters)
    norm = max(n_clusters - 1, 1)

    # Color arrays: original order and curated order
    orig_colors = cmap(labels / norm)[:, :3]  # (N, 3) indexed by original position
    curated_colors = cmap(labels[order] / norm)[:, :3]  # (N, 3) in curated order

    # For each original index, at which placement step is it picked?
    pick_step = np.full(N, N, dtype=np.int64)
    for i, idx in enumerate(order):
        pick_step[idx] = i

    # Pre-compute window info for curated order
    _, window_divs, window_bounds = compute_window_info(
        order, labels, token_counts, seq_len
    )

    # Original-order window diversities for reference line
    _, orig_divs, _ = compute_window_info(np.arange(N), labels, token_counts, seq_len)
    orig_mean_div = orig_divs.mean()

    seq_label = f"{seq_len // 1024}K" if seq_len >= 1024 else str(seq_len)

    log.info(f"Generating {n_frames} frames ({samples_per_frame} samples/frame)...")
    t0 = time.time()

    for frame_idx in range(n_frames):
        n_placed = min((frame_idx + 1) * samples_per_frame, N)

        # --- Build unified bar: [curated (bright) | remaining original (dimmed)] ---
        bar = np.ones((1, N, 3))  # white background

        # Left: placed items in curated order (full brightness)
        bar[0, :n_placed, :] = curated_colors[:n_placed]

        # Right: unpicked items in their original order (dimmed)
        not_picked = pick_step >= n_placed  # boolean mask over original indices
        remaining_colors = orig_colors[not_picked] * 0.4 + 0.6
        n_remaining = remaining_colors.shape[0]
        bar[0, n_placed : n_placed + n_remaining, :] = remaining_colors

        # --- Window stats ---
        completed_windows = np.searchsorted(window_bounds, n_placed, side="right")
        divs_so_far = window_divs[:completed_windows]

        if n_placed < N:
            current_window_clusters = set()
            window_start = (
                int(window_bounds[completed_windows - 1])
                if completed_windows > 0
                else 0
            )
            for i in range(window_start, n_placed):
                current_window_clusters.add(labels[order[i]])
            current_div = len(current_window_clusters)
        else:
            current_div = int(window_divs[-1]) if len(window_divs) > 0 else 0

        running_mean = divs_so_far.mean() if len(divs_so_far) > 0 else 0
        pct = n_placed / N * 100

        # --- Create figure: unified bar + diversity chart ---
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(fig_width, fig_height),
            gridspec_kw={"height_ratios": [1, 2.5]},
        )

        fig.suptitle(
            f"Diversity-Ordered Training ({n_clusters} clusters, {seq_label}-token windows)",
            fontsize=11,
            fontweight="bold",
            y=0.98,
        )

        # Unified bar
        ax_bar = axes[0]
        ax_bar.imshow(bar, aspect="auto", interpolation="none", extent=[0, N, 0, 1])

        # Divider cursor: white outline + dark center for visibility
        ax_bar.axvline(n_placed, color="white", linewidth=2.5, alpha=0.9)
        ax_bar.axvline(n_placed, color="black", linewidth=0.8, alpha=0.6)

        # Labels on each half
        if n_placed > N * 0.08:
            ax_bar.text(
                n_placed * 0.5,
                0.5,
                "Curated \u2192",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
                alpha=0.7,
            )
        if n_placed < N * 0.92:
            ax_bar.text(
                n_placed + (N - n_placed) * 0.5,
                0.5,
                "\u2190 Original",
                ha="center",
                va="center",
                fontsize=8,
                color="gray",
                fontweight="bold",
                alpha=0.5,
            )

        ax_bar.set_title(
            f"Placed: {n_placed:,} / {N:,} ({pct:.0f}%)  |  "
            f"Windows: {completed_windows} / {len(window_divs)}  |  "
            f"Current: {current_div} / {n_clusters}  |  "
            f"Mean: {running_mean:.1f}",
            fontsize=9,
            fontweight="bold",
        )
        ax_bar.set_yticks([])
        ax_bar.set_xticks([])

        # Diversity line chart
        ax = axes[1]
        if len(divs_so_far) > 1:
            ax.fill_between(
                range(len(divs_so_far)),
                divs_so_far,
                alpha=0.3,
                color="tab:blue",
            )
            ax.plot(
                divs_so_far,
                color="tab:blue",
                linewidth=1,
                alpha=0.8,
            )
        elif len(divs_so_far) == 1:
            ax.bar(0, divs_so_far[0], color="tab:blue", alpha=0.5, width=1)

        # Reference lines
        ax.axhline(n_clusters, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.axhline(
            orig_mean_div, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.5
        )
        ax.text(
            len(window_divs) * 0.98,
            orig_mean_div + 0.5,
            f"Original mean ({orig_mean_div:.1f})",
            fontsize=7,
            color="tab:red",
            ha="right",
            alpha=0.7,
        )

        ax.set_xlim(0, len(window_divs))
        ax.set_ylim(0, n_clusters + 2)
        ax.set_xlabel("Training sequence index", fontsize=9)
        ax.set_ylabel("Clusters\nper window", fontsize=8)
        ax.set_title(
            f"Window Diversity ({seq_label}-token windows)",
            fontsize=9,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(
            os.path.join(frame_dir, f"frame_{frame_idx:04d}.png"),
            dpi=120,
            bbox_inches="tight",
        )
        plt.close(fig)

        if (frame_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            fps = (frame_idx + 1) / elapsed
            log.info(f"  Frame {frame_idx + 1}/{n_frames} ({fps:.1f} frames/s)")

    elapsed = time.time() - t0
    log.info(
        f"  Done: {n_frames} frames in {elapsed:.1f}s ({n_frames / elapsed:.1f} fps)"
    )


def generate_umap_frames(
    labels,
    order,
    token_counts,
    umap_2d,
    n_clusters,
    seq_len,
    frame_dir,
    n_frames=600,
):
    """Generate UMAP cloud animation frames.

    Points start as gray dots and light up in their cluster color as the
    stratified greedy algorithm picks them.  Uses direct pixel painting
    for speed — only the newly picked points are colored each frame.
    """
    N = len(order)
    samples_per_frame = max(1, N // n_frames)

    cmap = matplotlib.colormaps["tab20"].resampled(n_clusters)
    norm = max(n_clusters - 1, 1)

    # Per-point cluster colors (RGB float32)
    cluster_colors = cmap(labels / norm)[:, :3].astype(np.float32)

    # Map UMAP coordinates to pixel grid
    res = 800
    margin = 0.04
    x, y = umap_2d[:, 0], umap_2d[:, 1]
    x_lo, x_hi = x.min(), x.max()
    y_lo, y_hi = y.min(), y.max()
    x_span = x_hi - x_lo
    y_span = y_hi - y_lo

    px = (x - x_lo + margin * x_span) / ((1 + 2 * margin) * x_span) * (res - 1)
    py = (y - y_lo + margin * y_span) / ((1 + 2 * margin) * y_span) * (res - 1)
    px = np.clip(px.astype(np.int32), 1, res - 2)
    py = np.clip(py.astype(np.int32), 1, res - 2)
    py = res - 1 - py  # flip y for image coordinates

    # For each original index, which step picks it?
    pick_step = np.full(N, N, dtype=np.int64)
    for i, idx in enumerate(order):
        pick_step[idx] = i

    # Pre-compute window info
    _, window_divs, window_bounds = compute_window_info(
        order, labels, token_counts, seq_len
    )
    _, orig_divs, _ = compute_window_info(np.arange(N), labels, token_counts, seq_len)
    orig_mean_div = orig_divs.mean()

    seq_label = f"{seq_len // 1024}K" if seq_len >= 1024 else str(seq_len)

    # 3x3 pixel block offsets for each point
    offsets = [(dy, dx) for dy in range(-1, 2) for dx in range(-1, 2)]

    unpicked_color = np.float32([0.85, 0.85, 0.85])

    log.info(
        f"Generating {n_frames} UMAP frames ({samples_per_frame} samples/frame)..."
    )
    t0 = time.time()

    # Initialize image: white background + all points as gray
    img = np.ones((res, res, 3), dtype=np.float32)
    for dy, dx in offsets:
        img[py + dy, px + dx] = unpicked_color

    # Create figure once and reuse
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(img, interpolation="none", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])

    title = fig.suptitle("", fontsize=11, fontweight="bold", y=0.98)
    stats_box = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray"
        ),
        family="monospace",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    prev_n = 0
    for frame_idx in range(n_frames):
        n_placed = min((frame_idx + 1) * samples_per_frame, N)

        # Incrementally color only the newly picked points
        new_mask = (pick_step >= prev_n) & (pick_step < n_placed)
        new_colors = cluster_colors[new_mask]
        new_py = py[new_mask]
        new_px = px[new_mask]
        for dy, dx in offsets:
            img[new_py + dy, new_px + dx] = new_colors

        prev_n = n_placed

        # Window stats
        completed_windows = np.searchsorted(window_bounds, n_placed, side="right")
        divs_so_far = window_divs[:completed_windows]
        running_mean = divs_so_far.mean() if len(divs_so_far) > 0 else 0
        pct = n_placed / N * 100

        n_active = len(set(labels[order[:n_placed]])) if n_placed > 0 else 0

        # Update figure elements (no figure recreation)
        im.set_data(img)
        title.set_text(
            f"Diversity-Ordered Sampling ({n_clusters} clusters, {seq_label} windows)"
        )
        stats_box.set_text(
            f"Placed:  {n_placed:>6,} / {N:,} ({pct:>3.0f}%)\n"
            f"Clusters: {n_active:>3} / {n_clusters}\n"
            f"Windows: {completed_windows:>5,} / {len(window_divs):,}\n"
            f"Mean div: {running_mean:>5.1f} / {n_clusters}"
        )

        fig.savefig(
            os.path.join(frame_dir, f"frame_{frame_idx:04d}.png"),
            dpi=100,
            bbox_inches="tight",
        )

        if (frame_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            fps = (frame_idx + 1) / elapsed
            log.info(f"  Frame {frame_idx + 1}/{n_frames} ({fps:.1f} frames/s)")

    plt.close(fig)
    elapsed = time.time() - t0
    log.info(
        f"  Done: {n_frames} frames in {elapsed:.1f}s ({n_frames / elapsed:.1f} fps)"
    )


def encode_video(frame_dir, output, fps):
    """Stitch PNG frames into an mp4 with ffmpeg."""
    frame_pattern = os.path.join(frame_dir, "frame_%04d.png")
    log.info(f"Encoding video with ffmpeg ({fps} fps)...")
    cmd = (
        f"ffmpeg -y -framerate {fps} "
        f"-i {frame_pattern} "
        f"-c:v libx264 -pix_fmt yuv420p -crf 18 "
        f"-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' "
        f"{output}"
    )
    ret = os.system(cmd)
    if ret == 0:
        size_mb = os.path.getsize(output) / 1e6
        log.info(f"  Saved: {output} ({size_mb:.1f} MB)")
    else:
        log.error(f"  ffmpeg failed with code {ret}")


def main():
    parser = argparse.ArgumentParser(description="Animate diversity ordering process")
    parser.add_argument("--embeddings", required=True, help="Embeddings .npy file")
    parser.add_argument("--token-counts", required=True, help="Token counts .npy file")
    parser.add_argument("--umap", default=None, help="UMAP 2D coordinates .npy file")
    parser.add_argument("--n-clusters", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-frames", type=int, default=600)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--output", default="ordering_animation.mp4", help="Output video file"
    )
    args = parser.parse_args()

    log.info("Loading data...")
    embeddings = np.load(args.embeddings)
    token_counts = np.load(args.token_counts)
    log.info(f"  {len(embeddings):,} samples, dim={embeddings.shape[1]}")

    log.info("Clustering and ordering...")
    labels, order = cluster_and_order(embeddings, args.n_clusters, args.seed)

    # Bar animation
    with tempfile.TemporaryDirectory() as frame_dir:
        generate_frames(
            labels,
            order,
            token_counts,
            args.n_clusters,
            args.seq_len,
            frame_dir,
            args.n_frames,
        )
        encode_video(frame_dir, args.output, args.fps)

    # UMAP cloud animation (if coordinates provided)
    if args.umap:
        umap_2d = np.load(args.umap)
        log.info(f"Loaded UMAP 2D: {umap_2d.shape}")

        base, ext = os.path.splitext(args.output)
        umap_output = f"{base}_umap{ext}"

        with tempfile.TemporaryDirectory() as frame_dir:
            generate_umap_frames(
                labels,
                order,
                token_counts,
                umap_2d,
                args.n_clusters,
                args.seq_len,
                frame_dir,
                args.n_frames,
            )
            encode_video(frame_dir, umap_output, args.fps)


if __name__ == "__main__":
    main()
