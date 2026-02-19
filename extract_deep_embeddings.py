#!/usr/bin/env python3
"""
Extract last-layer hidden state embeddings from GPT-OSS using the torchtitan
model with FlexAttention + attention sinks.

Uses the SAME model implementation as the training pipeline — including:
  - FlexAttention with compiled block masks
  - Alternating sliding-window (128 tokens, even layers) / full-causal (odd layers)
  - Learned per-head attention sinks (LSE rescaling)
  - YaRN RoPE for extended context

This produces numerically correct embeddings that reflect the actual attention
patterns the model uses during training, rather than the generic SDPA that HF
AutoModel uses (which ignores sinks entirely).

The embeddings are L2-normalized and saved as a (N, D) float32 numpy array,
ready to be loaded by curate_dataset.py via --load-embeddings.

IMPORTANT: Requires the torchtitan venv with PyTorch nightly (for flex_attention).
    source ~/git/torchtitan/venv312/bin/activate

Usage:
    # Full run (69K docs, batch_size=1 recommended for variable lengths):
    python extract_deep_embeddings.py \
        --input /mnt/models/persona_datasets/persona-iota-v2.jsonl \
        --output deep_embeddings.npy \
        --model-dir /mnt/models/openai/gpt-oss-20b \
        --batch-size 1

    # With tensor parallelism (halves per-GPU memory, handles longer docs):
    torchrun --nproc_per_node=2 extract_deep_embeddings.py \
        --input persona-iota.jsonl \
        --output deep_embeddings.npy \
        --model-dir /mnt/models/openai/gpt-oss-20b \
        --tp 2

    # With truncation (for very long documents):
    python extract_deep_embeddings.py \
        --input persona-iota.jsonl \
        --output deep_embeddings_8k.npy \
        --model-dir /mnt/models/openai/gpt-oss-20b \
        --max-tokens 8192

    # Validation subset:
    python extract_deep_embeddings.py \
        --input persona-iota.jsonl \
        --output deep_50.npy \
        --model-dir /mnt/models/openai/gpt-oss-20b \
        --limit 50
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

# Add torchtitan to path for model imports
TORCHTITAN_ROOT = os.environ.get("TORCHTITAN_ROOT", "/mnt/git/torchtitan.temp")
sys.path.insert(0, TORCHTITAN_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def setup_distributed(tp_degree: int) -> tuple:
    """Initialize distributed process group and build TP mesh.

    Returns (rank, world_size, parallel_dims).
    """
    from torchtitan.distributed.parallel_dims import ParallelDims

    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    assert world_size == tp_degree, (
        f"World size ({world_size}) must equal TP degree ({tp_degree}). "
        f"Launch with: torchrun --nproc_per_node={tp_degree}"
    )
    torch.cuda.set_device(rank)

    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=tp_degree,
        pp=1,
        ep=1,
        etp=1,
        world_size=tp_degree,
    )
    parallel_dims.build_mesh()
    log.info(f"Rank {rank}/{world_size}: TP mesh built")
    return rank, world_size, parallel_dims


def load_texts(input_path: str, text_field: str = "text", limit: int = 0) -> list[str]:
    """Load document texts from a JSONL file."""
    texts = []
    with open(input_path) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get(text_field)
            if text is None:
                raise KeyError(
                    f"Field '{text_field}' not found at line {lineno}. "
                    f"Available: {list(obj.keys())}"
                )
            texts.append(text)
            if limit and len(texts) >= limit:
                break
    return texts


def load_torchtitan_model(
    model_dir: str,
    device: str = "cuda:0",
    model_flavor: str = "20b",
    parallel_dims=None,
) -> tuple:
    """
    Load GPT-OSS using the torchtitan model with FlexAttention + sinks.

    This loads the same model implementation used during training, including
    attention sinks and alternating sliding-window/full-causal masks.

    Args:
        parallel_dims: If provided with tp_enabled, applies tensor parallelism
            via apply_non_moe_tp + apply_moe_ep_tp after loading weights.

    Returns (model, tokenizer, model_args).
    """
    from transformers import AutoTokenizer

    from torchtitan.models.gpt_oss import gptoss_configs
    from torchtitan.models.gpt_oss.model.model import GptOssModel, precompute_rope_cache
    from torchtitan.models.gpt_oss.model.state_dict_adapter import (
        GptOssStateDictAdapter,
    )

    log.info(f"Loading torchtitan {model_flavor} model from {model_dir}...")
    t0 = time.time()

    # 1. Tokenizer (same as training pipeline)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    log.info(f"  Tokenizer loaded (vocab={tokenizer.vocab_size})")

    # 2. Model args from registered config
    model_args = gptoss_configs[model_flavor]

    # 3. Create model on meta device (no memory allocation)
    with torch.device("meta"):
        model = GptOssModel(model_args)

    # 4. Load HF safetensors with MXFP4 dequantization
    adapter = GptOssStateDictAdapter(model_args, model_dir)
    state_dict = adapter.load_hf_safetensors_direct(
        model_dir, target_dtype=torch.bfloat16
    )

    # Drop lm_head — we only need hidden states, not logits.
    # BUT: when TP is enabled, apply_non_moe_tp parallelizes output.weight
    # via parallelize_module, so the parameter must exist even though we
    # never call model.output() in our forward pass.
    tp_enabled = parallel_dims is not None and parallel_dims.tp_enabled
    if tp_enabled:
        log.info(f"  State dict: {len(state_dict)} tensors (keeping lm_head for TP)")
    else:
        state_dict.pop("output.weight", None)
        log.info(f"  State dict: {len(state_dict)} tensors (lm_head dropped)")

    # 5. Materialize from meta to CPU (allocates uninitialized tensors),
    #    then load real weights. This avoids the ~80 GiB spike of creating
    #    a full model on CPU with random init.
    model.to_empty(device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, assign=True, strict=False)
    if unexpected:
        log.warning(f"  Unexpected keys: {unexpected}")
    expected_missing = set() if tp_enabled else {"output.weight"}
    real_missing = set(missing) - expected_missing
    if real_missing:
        log.error(f"  Missing keys (non-lm_head): {real_missing}")
        raise RuntimeError(f"Missing model weights: {real_missing}")
    del state_dict

    # 6. Recompute rope_cache (was a meta tensor from meta-device init)
    model.rope_cache = precompute_rope_cache(
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_factor,
        model_args.ntk_alpha,
        model_args.ntk_beta,
        model_args.original_seq_len,
    )

    # 7. Use compiled FlexAttention with basic options (no max_autotune).
    #    The Triton kernel uses block-sparse computation — O(seq) memory for
    #    sliding window, vs O(seq²) for unfused. This is critical for docs
    #    over ~13K tokens. We avoid max_autotune/coordinate_descent_tuning
    #    which fail on some GPU architectures (RTX PRO 6000 Blackwell).
    from torch.nn.attention.flex_attention import flex_attention as _raw_flex_attn
    from torchtitan.models.attention import FlexAttentionWrapper

    FlexAttentionWrapper._compiled_flex_attn = staticmethod(
        torch.compile(_raw_flex_attn)
    )

    # 8. Move to GPU (full model temporarily, ~40 GiB for 20B)
    model = model.to(device)
    model.eval()

    # 9. Apply tensor parallelism (shards model in-place, ~40→20 GiB per rank)
    if tp_enabled:
        from torchtitan.models.gpt_oss.infra.parallelize import (
            apply_moe_ep_tp,
            apply_non_moe_tp,
        )

        tp_mesh = parallel_dims.get_mesh("tp")
        apply_non_moe_tp(
            model,
            tp_mesh,
            loss_parallel=False,
            enable_float8_tensorwise_tp=False,
            enable_async_tp=False,
        )
        apply_moe_ep_tp(
            model,
            tp_mesh=tp_mesh,
            ep_mesh=None,
            ep_etp_mesh=None,
            etp_enabled=False,
        )
        log.info(f"  Applied TP={parallel_dims.tp} to model")
        # Release allocator's reserved-but-unused memory from loading the
        # full checkpoint (40 GiB) before sharding to 19.5 GiB per rank.
        # Without this, mem_get_info() reports ~15 GiB free instead of ~76 GiB.
        torch.cuda.empty_cache()

    mem_gb = torch.cuda.memory_allocated(device) / 1024**3
    log.info(
        f"  Model loaded in {time.time() - t0:.1f}s "
        f"(dim={model_args.dim}, {model_args.n_layers} layers, "
        f"GPU: {mem_gb:.1f} GiB)"
    )

    return model, tokenizer, model_args


def create_attention_masks(seq_len: int, model_args, device: str = "cuda:0"):
    """
    Create the FlexAttention block masks for a given sequence length.

    Returns {"basic_mask": ..., "sliding_window_mask": ...} matching
    what GptOssModel.get_attention_masks() produces.
    """
    from torch.nn.attention.flex_attention import and_masks, create_block_mask

    from torchtitan.models.attention import (
        get_causal_mask_mod,
        get_sliding_window_mask_mod,
    )

    causal = get_causal_mask_mod()
    sliding = get_sliding_window_mask_mod(model_args.sliding_window_size)

    # B=1 for causal mode (broadcasts across batch).
    # _compile=True evaluates the mask at block granularity ((seq/128)²
    # evaluations) instead of materializing a full [seq, seq] bool tensor
    # (seq² bytes — 9.6 GiB at 98K tokens). Critical for long sequences.
    basic_mask = create_block_mask(
        and_masks(causal), 1, None, seq_len, seq_len, _compile=True
    )
    sliding_window_mask = create_block_mask(
        and_masks(causal, sliding), 1, None, seq_len, seq_len, _compile=True
    )

    return {"basic_mask": basic_mask, "sliding_window_mask": sliding_window_mask}


def forward_hidden_states(
    model,
    tokens: torch.Tensor,
    attention_masks: dict,
) -> torch.Tensor:
    """
    Run forward pass through embedding + all transformer layers + norm,
    returning the last-layer hidden states (skipping lm_head).

    This is the same as GptOssModel.forward() but stops before self.output().
    """
    h = model.tok_embeddings(tokens)
    for layer in model.layers.values():
        h = layer(h, model.rope_cache, attention_masks)
    h = model.norm(h)
    # With TP, hidden states after norm are Shard(1) DTensors.
    # All-gather to get full [batch, seq, dim] on every rank.
    if isinstance(h, DTensor):
        h = h.full_tensor()
    return h  # (batch, seq_len, dim)


def extract_embeddings(
    texts: list[str],
    model_dir: str,
    max_tokens: int = 0,
    batch_size: int = 1,
    device: str = "cuda:0",
    start_idx: int = 0,
    output_path: str | None = None,
    model_flavor: str = "20b",
    reverse_order: bool = False,
    parallel_dims=None,
    rank: int = 0,
) -> np.ndarray:
    """
    Extract mean-pooled last-layer hidden state embeddings using the
    torchtitan model with FlexAttention + attention sinks.
    """
    tp_enabled = parallel_dims is not None and parallel_dims.tp_enabled
    model, tokenizer, model_args = load_torchtitan_model(
        model_dir,
        device,
        model_flavor,
        parallel_dims=parallel_dims,
    )
    hidden_dim = model_args.dim
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    n = len(texts)
    all_embeddings = np.zeros((n, hidden_dim), dtype=np.float32)

    # Resume from partial checkpoint if available (rank 0 only loads, all ranks use count)
    partial_path = (output_path + ".partial.npy") if output_path else None
    resumed_count = 0
    if rank == 0 and partial_path and os.path.exists(partial_path):
        partial = np.load(partial_path, allow_pickle=False)
        if partial.shape == all_embeddings.shape:
            all_embeddings = partial
            resumed_count = int(np.count_nonzero(np.any(partial != 0, axis=1)))
            log.info(
                f"  Resumed from partial checkpoint: {resumed_count}/{n} docs already done"
            )
            del partial
        else:
            log.warning(
                f"  Partial checkpoint shape {partial.shape} != expected {all_embeddings.shape}, ignoring"
            )
    # Broadcast resume state so all TP ranks agree on skip logic.
    # Rank 0 has the real data; broadcast a per-doc done mask (handles OOM gaps).
    done_mask = np.any(all_embeddings != 0, axis=1)  # shape (n,)
    if tp_enabled:
        rc_tensor = torch.tensor(resumed_count, device=device)
        dist.broadcast(rc_tensor, src=0)
        resumed_count = rc_tensor.item()
        done_tensor = torch.tensor(done_mask, dtype=torch.bool, device=device)
        dist.broadcast(done_tensor, src=0)
        done_mask = done_tensor.cpu().numpy()

    # Pre-tokenize all texts to get lengths for sorted batching.
    # Cache token IDs to disk (JSON) to skip re-tokenization on resume runs.
    token_cache_path = (output_path + ".tokens.json") if output_path else None
    cache_key = {
        "n": n,
        "max_tokens": max_tokens,
        "text_hash": hashlib.sha256(
            json.dumps([len(t) for t in texts]).encode()
            + texts[0][:200].encode()
            + texts[-1][:200].encode()
        ).hexdigest(),
    }
    all_token_ids = None
    if token_cache_path and os.path.exists(token_cache_path):
        try:
            with open(token_cache_path) as f:
                cached_data = json.load(f)
            if cached_data.get("key") == cache_key:
                all_token_ids = cached_data["token_ids"]
                log.info(f"  Loaded cached tokenization ({n} docs)")
            else:
                log.info("  Token cache invalidated (data changed), re-tokenizing")
        except Exception:
            log.info("  Token cache corrupted, re-tokenizing")

    # Also check for legacy pickle cache and migrate
    legacy_cache = (output_path + ".tokens.pkl") if output_path else None
    if all_token_ids is None and legacy_cache and os.path.exists(legacy_cache):
        try:
            with open(legacy_cache, "rb") as f:
                cached_data = pickle.load(f)
            if cached_data.get("key") == cache_key:
                all_token_ids = cached_data["token_ids"]
                log.info(
                    f"  Loaded legacy pickle cache ({n} docs), will migrate to JSON"
                )
        except Exception:
            pass

    if all_token_ids is None:
        log.info(f"Pre-tokenizing {n} documents...")
        t_tok = time.time()
        all_token_ids = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if max_tokens > 0 and len(ids) > max_tokens:
                ids = ids[:max_tokens]
            all_token_ids.append(ids)
        log.info(f"  Tokenized in {time.time() - t_tok:.1f}s")
        # Save cache as JSON (rank 0 only to avoid race)
        if token_cache_path and rank == 0:
            with open(token_cache_path, "w") as f:
                json.dump({"key": cache_key, "token_ids": all_token_ids}, f)
            log.info(f"  Saved token cache: {token_cache_path}")

    lengths = [len(ids) for ids in all_token_ids]
    log.info(
        f"  Lengths: min={min(lengths)}, median={sorted(lengths)[n // 2]}, "
        f"mean={sum(lengths) / n:.0f}, max={max(lengths)}"
    )
    if max_tokens > 0:
        truncated = sum(1 for length in lengths if length == max_tokens)
        if truncated:
            log.info(f"  {truncated} documents truncated to {max_tokens} tokens")

    # Sort by length for efficient batching (minimizes padding waste)
    sorted_indices = sorted(range(n), key=lambda i: lengths[i])
    if reverse_order:
        sorted_indices = sorted_indices[::-1]
        log.info("  Processing in REVERSE order (longest first)")

    # Precompute cumulative token counts for efficient progress reporting
    sorted_lengths = np.array([lengths[i] for i in sorted_indices])
    cumulative_tokens = np.cumsum(sorted_lengths)

    # Cache for attention masks by sequence length (avoid re-creating per batch)
    mask_cache: dict[int, dict] = {}

    # Dynamic batching: compute memory budget for activations.
    # With compiled FlexAttention (block-sparse Triton kernel), per-sample
    # activation memory scales as O(seq_len * dim). Empirically ~15 bytes
    # per token covers hidden states + Q/K/V + MoE intermediates per layer.
    model_mem = torch.cuda.memory_allocated(device)
    gpu_total = torch.cuda.get_device_properties(device).total_memory
    original_activation_budget = int((gpu_total - model_mem) * 0.70)
    activation_budget = original_activation_budget
    bytes_per_token = model_args.dim * 15  # ~43 KiB/token for dim=2880

    if rank == 0:
        log.info(
            f"  Dynamic batching: {activation_budget / 1e9:.1f} GiB activation budget, "
            f"max_batch_size={batch_size}"
        )

    log.info(
        f"Extracting embeddings for {n} documents "
        f"(max_tokens={'unlimited' if max_tokens == 0 else max_tokens}, "
        f"dynamic_batch (max={batch_size}), FlexAttention + sinks)..."
    )
    t0 = time.time()
    docs_done = 0
    oom_skipped: list[int] = []

    batch_start = 0
    while batch_start < n:
        # Compute dynamic batch size based on longest doc in this group.
        # Docs are sorted by length, so with --reverse the longest is at
        # batch_start; without, it's at the far end of the batch window.
        if reverse_order:
            longest_in_window = lengths[sorted_indices[batch_start]]
        else:
            window_end = min(batch_start + batch_size, n)
            longest_in_window = lengths[sorted_indices[window_end - 1]]

        per_sample_bytes = longest_in_window * bytes_per_token
        dynamic_bs = max(1, activation_budget // per_sample_bytes)
        dynamic_bs = min(dynamic_bs, batch_size, n - batch_start)

        batch_idx = sorted_indices[batch_start : batch_start + dynamic_bs]
        actual_bs = len(batch_idx)

        # Skip docs already completed (from partial checkpoint resume)
        if resumed_count > 0 and all(done_mask[i] for i in batch_idx):
            docs_done += actual_bs
            batch_start += dynamic_bs
            continue

        # Get pre-tokenized IDs for this batch
        batch_ids = [all_token_ids[i] for i in batch_idx]

        # Pad to longest in batch
        max_len = max(len(ids) for ids in batch_ids)

        # SequenceParallel shards the sequence dim across TP ranks.
        # Each rank gets seq_len / tp tokens, but the split must be even.
        # torchtitan enforces seq_len % (tp * 2) == 0 via seq_len_divisor.
        if tp_enabled:
            divisor = parallel_dims.tp * 2  # = 4 for TP=2
            max_len = ((max_len + divisor - 1) // divisor) * divisor

        padded = torch.full(
            (actual_bs, max_len),
            pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(actual_bs, max_len, dtype=torch.long)
        for j, ids in enumerate(batch_ids):
            padded[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[j, : len(ids)] = 1

        padded = padded.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        # Evict mask cache if GPU memory is getting low. Each cached
        # BlockMask holds O((seq/block)^2) data — hundreds of entries
        # from variable-length docs can fill GPU memory.
        if max_len not in mask_cache and len(mask_cache) > 2:
            free_phys = torch.cuda.mem_get_info(device)[0]
            alloc_free = torch.cuda.memory_reserved(
                device
            ) - torch.cuda.memory_allocated(device)
            free_mem = free_phys + alloc_free
            if free_mem < 20 * 1024**3:  # < 20 GiB effective free
                mask_cache.clear()
                torch.cuda.empty_cache()

        # With TP, an OOM during forward_hidden_states() would deadlock
        # (one rank exits a collective while the other waits). Pre-check
        # available memory and skip batches that are too large. All ranks
        # process the same batch, so the decision is deterministic.
        #
        # With compiled FlexAttention (Triton kernel), memory is O(seq) per
        # layer (block-sparse), not O(seq²). The main costs are:
        #   - Per-layer activations: ~seq × dim × bf16 × 3 (Q,K,V + intermediates)
        #   - MoE intermediates: ~seq × inter × bf16 / tp
        #   - Block masks: O((seq/block)²) — small
        # Conservative estimate: 24 layers × seq × dim × 20 bytes per layer
        # (covers Q/K/V projections, MoE intermediates, temporary buffers).
        if tp_enabled and max_len > 8192:
            # Effective free = physical_free + allocator's unused pool.
            # mem_get_info() alone misses the reserved-but-unallocated pool
            # (~50+ GiB after TP sharding), drastically underreporting.
            free_phys = torch.cuda.mem_get_info(device)[0]
            reserved = torch.cuda.memory_reserved(device)
            allocated = torch.cuda.memory_allocated(device)
            free_mem = free_phys + (reserved - allocated)
            # With compiled FlexAttention, per-layer peak is modest.
            # Use heuristic: seq × dim × 20 bytes (covers one layer's
            # Q/K/V/intermediates/MoE with headroom).
            estimated_bytes = int(max_len * model_args.dim * 20)
            if estimated_bytes > free_mem * 0.85:
                skipped = list(batch_idx)
                oom_skipped.extend(skipped)
                if rank == 0:
                    log.warning(
                        f"Skipping batch (max_len={max_len:,}): estimated "
                        f"{estimated_bytes / 1e9:.1f} GiB > 85% of free "
                        f"{free_mem / 1e9:.1f} GiB. Would deadlock with TP."
                    )
                docs_done += actual_bs
                batch_start += dynamic_bs
                continue

        try:
            # Get or create attention masks for this sequence length
            if max_len not in mask_cache:
                mask_cache[max_len] = create_attention_masks(
                    max_len, model_args, device
                )
            masks = mask_cache[max_len]

            with torch.inference_mode():
                hidden_states = forward_hidden_states(model, padded, masks)

                # Upcast to fp32 BEFORE accumulation. bf16 has ~3 digits of
                # precision — summing thousands of values in bf16 causes
                # catastrophic rounding.
                hidden_f32 = hidden_states.float()
                del hidden_states

                # Mean-pool over non-padding positions
                pool_mask = attention_mask.unsqueeze(-1).float()
                summed = (hidden_f32 * pool_mask).sum(dim=1)
                del hidden_f32
                count = pool_mask.sum(dim=1).clamp(min=1)
                mean_embed = summed / count

                # L2 normalize
                mean_embed = torch.nn.functional.normalize(mean_embed, dim=1)

                embeddings_batch = mean_embed.cpu().numpy()

            oom = torch.tensor(0, device=device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            mask_cache.clear()
            oom = torch.tensor(1, device=device)
            embeddings_batch = None

        # With TP, all ranks must agree on skip vs proceed (otherwise NCCL hangs)
        if tp_enabled:
            dist.all_reduce(oom, op=dist.ReduceOp.MAX)

        if oom.item():
            if actual_bs > 1:
                # OOM with batch>1: reduce activation budget and retry
                # (the while loop will recompute a smaller dynamic_bs)
                activation_budget = int(activation_budget * 0.6)
                if rank == 0:
                    log.warning(
                        f"OOM on batch with max_len={max_len}, bs={actual_bs}. "
                        f"Reducing budget to {activation_budget / 1e9:.1f} GiB and retrying..."
                    )
                continue  # retry same batch_start with smaller budget
            else:
                # OOM with batch=1: truly can't process this doc
                oom_skipped.extend(list(batch_idx))
                if rank == 0:
                    log.warning(
                        f"OOM on batch with max_len={max_len}, bs=1. "
                        f"Skipping doc {batch_idx[0]}. "
                        f"Completed {docs_done}/{n} docs so far."
                    )
                docs_done += actual_bs
                batch_start += dynamic_bs
                continue

        # Free activations eagerly
        del padded, attention_mask, pool_mask, summed, count, mean_embed

        # Gradually recover activation budget after successful batches
        if activation_budget < original_activation_budget:
            activation_budget = min(
                int(activation_budget * 1.1), original_activation_budget
            )

        # Store in original order
        for j, orig_idx in enumerate(batch_idx):
            all_embeddings[orig_idx] = embeddings_batch[j]

        docs_done += actual_bs

        # Periodic checkpoint (every ~5000 docs) — rank 0 only
        if (
            rank == 0
            and output_path
            and docs_done % 5000 < dynamic_bs
            and docs_done > 0
        ):
            partial = output_path + ".partial.npy"
            # Merge with existing partial (other worker may have updated it)
            if os.path.exists(partial):
                try:
                    existing = np.load(partial, allow_pickle=False)
                    if existing.shape == all_embeddings.shape:
                        # Keep whichever row is non-zero (merge both workers)
                        mask = np.any(existing != 0, axis=1) & ~np.any(
                            all_embeddings != 0, axis=1
                        )
                        all_embeddings[mask] = existing[mask]
                    del existing
                except Exception:
                    pass  # stale file, just overwrite
            np.save(partial, all_embeddings)
            log.info(f"  Checkpoint saved: {partial} ({docs_done}/{n})")

        # Progress logging — rank 0 only
        if rank == 0 and (
            docs_done % max(100, dynamic_bs * 5) < dynamic_bs or docs_done == n
        ):
            elapsed = time.time() - t0
            rate = docs_done / elapsed if elapsed > 0 else 0
            eta = (n - docs_done) / rate if rate > 0 else 0
            tokens_done = int(cumulative_tokens[docs_done - 1])
            mem_gb = torch.cuda.memory_allocated(device) / 1024**3
            log.info(
                f"  [{start_idx + docs_done:>6d}] {docs_done:>6d}/{n} "
                f"(bs={dynamic_bs}, {rate:.1f} docs/s, "
                f"{tokens_done / elapsed:.0f} tok/s, "
                f"ETA {eta / 60:.1f}min, GPU {mem_gb:.1f}GiB)"
            )

        batch_start += dynamic_bs

    if rank == 0:
        elapsed = time.time() - t0
        total_tokens = sum(lengths)
        log.info(
            f"  Extraction complete in {elapsed:.1f}s "
            f"({n / elapsed:.1f} docs/s, {total_tokens / elapsed:.0f} tok/s, "
            f"{total_tokens:,} total tokens)"
        )
        if oom_skipped:
            log.warning(
                f"  {len(oom_skipped)} documents skipped due to OOM "
                f"(indices: {oom_skipped[:20]}{'...' if len(oom_skipped) > 20 else ''})"
            )

        # Mark zero-vector rows (OOM-skipped or degenerate) as NaN so
        # downstream clustering doesn't silently include them.
        zero_mask = ~np.any(all_embeddings != 0, axis=1)
        n_zero = int(zero_mask.sum())
        if n_zero > 0:
            log.warning(
                f"  {n_zero} documents have zero embeddings (OOM-skipped or degenerate). "
                f"Setting to NaN to prevent silent clustering corruption."
            )
            all_embeddings[zero_mask] = np.nan

    return all_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Extract deep embeddings using torchtitan GPT-OSS model "
        "with FlexAttention + attention sinks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output .npy file")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="HuggingFace model directory (with safetensors + tokenizer)",
    )
    parser.add_argument(
        "--model-flavor",
        type=str,
        default="20b",
        choices=["20b", "120b", "debugmodel"],
        help="Model flavor from torchtitan registry (default: 20b)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Max tokens per document (default: 0 = no truncation). "
        "FlexAttention with sliding window makes long sequences feasible: "
        "96K tokens uses ~65 GiB on a 96 GiB GPU (20B model). "
        "Truncation is rarely needed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Max documents per forward pass (default: 64). "
        "Actual batch size is dynamically computed based on available "
        "GPU memory and sequence length — small batches for long docs, "
        "large batches for short docs.",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU device index (ignored with --tp)"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel degree. Requires torchrun --nproc_per_node=N. "
        "TP=2 halves per-GPU memory (~40→20 GiB model + smaller activations).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=0,
        help="Hard cap on sequence length (0 = no limit). Docs longer than "
        "this are truncated. Unfused FlexAttention materializes a full "
        "[heads, seq, seq] score matrix; ~18K tokens is the safe max "
        "for TP=2 on 96 GiB GPUs.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Process documents longest-first (for dual-GPU: one worker shortest-first, one longest-first)",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="JSON field containing document text",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N documents (0 = all)",
    )
    parser.add_argument(
        "--torchtitan-root",
        type=str,
        default=None,
        help="Path to torchtitan repo (default: $TORCHTITAN_ROOT or /mnt/git/torchtitan.temp)",
    )
    args = parser.parse_args()

    # Override torchtitan path if specified
    if args.torchtitan_root:
        sys.path.insert(0, args.torchtitan_root)

    # Setup distributed if TP requested
    if args.tp > 1:
        rank, world_size, parallel_dims = setup_distributed(args.tp)
        device = f"cuda:{rank}"
    else:
        rank, parallel_dims = 0, None
        device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)

    # Load texts
    log.info(f"Loading texts from {args.input}")
    texts = load_texts(args.input, args.text_field, args.limit)
    n_total = len(texts)
    log.info(f"  {n_total:,} documents loaded")

    # Check for cached embeddings (incremental support)
    start_idx = 0
    cached = None
    if os.path.exists(args.output):
        cached = np.load(args.output, allow_pickle=False)
        if len(cached) == n_total:
            n_zero = int(np.sum(~np.any(cached != 0, axis=1)))
            if n_zero == 0:
                log.info(
                    f"Output already exists with {n_total} embeddings. Nothing to do."
                )
                if args.tp > 1:
                    dist.destroy_process_group()
                return
            else:
                log.info(
                    f"Output has {n_total} rows but {n_zero} are zero (skipped). "
                    f"Resuming to fill them..."
                )
        elif len(cached) < n_total:
            start_idx = len(cached)
            log.info(
                f"Found {len(cached)} cached embeddings, "
                f"extracting {n_total - len(cached)} new..."
            )
            texts = texts[start_idx:]
        else:
            log.warning(
                f"Cache has MORE rows ({len(cached)}) than input ({n_total}). "
                f"Re-extracting from scratch."
            )
            cached = None

    # Merge --max-seq-len into --max-tokens (both truncate, take the tighter)
    max_tokens = args.max_tokens
    if args.max_seq_len > 0:
        max_tokens = (
            args.max_seq_len if max_tokens == 0 else min(max_tokens, args.max_seq_len)
        )
    if max_tokens > 0 and rank == 0:
        log.info(f"Truncating documents to {max_tokens:,} tokens")

    # Extract
    embeddings = extract_embeddings(
        texts,
        args.model_dir,
        max_tokens=max_tokens,
        batch_size=args.batch_size,
        device=device,
        start_idx=start_idx,
        output_path=args.output,
        model_flavor=args.model_flavor,
        reverse_order=args.reverse,
        parallel_dims=parallel_dims,
        rank=rank,
    )

    # Only rank 0 saves and reports stats
    if rank == 0:
        # Concatenate with cache if incremental
        if cached is not None and start_idx > 0:
            embeddings = np.concatenate([cached, embeddings], axis=0)
            log.info(f"  Combined with cache: {embeddings.shape}")

        # Save
        np.save(args.output, embeddings)
        log.info(f"Saved: {args.output} {embeddings.shape}")

        # Summary stats
        norms = np.linalg.norm(embeddings, axis=1)
        log.info(
            f"  Norm stats: mean={norms.mean():.4f}, std={norms.std():.6f} "
            f"(should be ~1.0 for L2-normalized)"
        )

        # Quick self-similarity check (sample if large)
        n_sim = len(embeddings)
        if n_sim > 5000:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(n_sim, 5000, replace=False)
            sim_emb = embeddings[sample_idx]
            sim_label = " (5K sample)"
        else:
            sim_emb = embeddings
            sim_label = ""

        if len(sim_emb) > 1:
            sim = sim_emb @ sim_emb.T
            np.fill_diagonal(sim, 0)
            upper = sim[np.triu_indices(len(sim), k=1)]
            log.info(
                f"  Cosine similarity{sim_label}: mean={upper.mean():.4f}, "
                f"std={upper.std():.4f}, max={upper.max():.4f}"
            )

    # Cleanup distributed
    if args.tp > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
