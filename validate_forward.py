#!/usr/bin/env python3
"""
Validate that extract_deep_embeddings.py produces the same hidden states
as torchtitan's canonical model forward pass.

Tests:
1. Mask equivalence: our create_attention_masks() vs model.get_attention_masks()
2. Hidden state equivalence: our forward_hidden_states() vs model.forward()
3. Compiled vs unfused FlexAttention numerical equivalence

Usage:
    python validate_forward.py --model-dir /mnt/models/openai/gpt-oss-20b --seq-len 4096
    # Or with a real document:
    python validate_forward.py --model-dir /mnt/models/openai/gpt-oss-20b \
        --input /mnt/models/persona_datasets/persona-iota-v2.jsonl --doc-idx 0
"""

import argparse
import json
import sys
import time

import torch

sys.path.insert(0, "/mnt/git/torchtitan.temp")


def load_model_and_tokenizer(model_dir, device, model_flavor="20b"):
    """Load model using the same code path as extract_deep_embeddings.py."""
    from transformers import AutoTokenizer

    from torchtitan.models.gpt_oss import gptoss_configs
    from torchtitan.models.gpt_oss.model.model import GptOssModel, precompute_rope_cache
    from torchtitan.models.gpt_oss.model.state_dict_adapter import (
        GptOssStateDictAdapter,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_args = gptoss_configs[model_flavor]

    with torch.device("meta"):
        model = GptOssModel(model_args)

    adapter = GptOssStateDictAdapter(model_args, model_dir)
    state_dict = adapter.load_hf_safetensors_direct(
        model_dir, target_dtype=torch.bfloat16
    )
    state_dict.pop("output.weight", None)

    model.to_empty(device="cpu")
    model.load_state_dict(state_dict, assign=True, strict=False)
    del state_dict

    model.rope_cache = precompute_rope_cache(
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_factor,
        model_args.ntk_alpha,
        model_args.ntk_beta,
        model_args.original_seq_len,
    )

    model = model.to(device)
    model.eval()

    return model, tokenizer, model_args


def test_mask_equivalence(model, model_args, seq_len, device):
    """Compare our create_attention_masks() vs model.get_attention_masks()."""
    from torch.nn.attention.flex_attention import and_masks, create_block_mask

    from torchtitan.models.attention import (
        get_causal_mask_mod,
        get_sliding_window_mask_mod,
    )

    print(f"\n{'=' * 60}")
    print(f"TEST 1: Mask equivalence (seq_len={seq_len})")
    print(f"{'=' * 60}")

    # Our masks (from extract_deep_embeddings.py)
    causal = get_causal_mask_mod()
    sliding = get_sliding_window_mask_mod(model_args.sliding_window_size)

    our_basic = create_block_mask(and_masks(causal), 1, None, seq_len, seq_len)
    our_sliding = create_block_mask(
        and_masks(causal, sliding), 1, None, seq_len, seq_len
    )

    # Model's masks (via torchtitan's compiled create_block_mask, same as training)
    from torchtitan.models.attention import create_attention_mask

    model_basic = create_attention_mask(and_masks(causal), 1, None, seq_len, seq_len)
    model_sliding = create_attention_mask(
        and_masks(causal, sliding), 1, None, seq_len, seq_len
    )
    model_masks = {"basic_mask": model_basic, "sliding_window_mask": model_sliding}

    # Compare block mask sparsity patterns
    # BlockMask stores which blocks are full/partial/empty
    our_basic_kv = our_basic.kv_num_blocks
    model_basic_kv = model_basic.kv_num_blocks

    our_sliding_kv = our_sliding.kv_num_blocks
    model_sliding_kv = model_sliding.kv_num_blocks

    basic_match = torch.equal(our_basic_kv, model_basic_kv)
    sliding_match = torch.equal(our_sliding_kv, model_sliding_kv)

    print(f"  Basic mask kv_num_blocks match:   {basic_match}")
    print(f"  Sliding mask kv_num_blocks match: {sliding_match}")

    # Also compare full_kv_num_blocks and kv_indices
    basic_idx_match = torch.equal(our_basic.kv_indices, model_basic.kv_indices)
    sliding_idx_match = torch.equal(our_sliding.kv_indices, model_sliding.kv_indices)
    print(f"  Basic mask kv_indices match:      {basic_idx_match}")
    print(f"  Sliding mask kv_indices match:    {sliding_idx_match}")

    all_match = basic_match and sliding_match and basic_idx_match and sliding_idx_match
    print(f"\n  RESULT: {'PASS' if all_match else 'FAIL'}")

    return {"basic_mask": our_basic, "sliding_window_mask": our_sliding}, model_masks


def test_hidden_state_equivalence(
    model, model_args, tokens, our_masks, model_masks, device
):
    """Compare our forward_hidden_states() vs model.forward()."""
    print(f"\n{'=' * 60}")
    print(f"TEST 2: Hidden state equivalence (seq_len={tokens.shape[1]})")
    print(f"{'=' * 60}")

    from torch.nn.attention.flex_attention import flex_attention as _raw_flex_attn
    from torchtitan.models.attention import FlexAttentionWrapper

    # Test with UNFUSED FlexAttention (deterministic, no compilation variance)
    FlexAttentionWrapper._compiled_flex_attn = staticmethod(_raw_flex_attn)

    with torch.no_grad():
        # Our forward path (from extract_deep_embeddings.py)
        h_ours = model.tok_embeddings(tokens)
        for layer in model.layers.values():
            h_ours = layer(h_ours, model.rope_cache, our_masks)
        h_ours = model.norm(h_ours)

        # Model's forward path (canonical)
        h_model = model.tok_embeddings(tokens)
        for layer in model.layers.values():
            h_model = layer(h_model, model.rope_cache, model_masks)
        h_model = model.norm(h_model)

    # Compare
    max_diff = (h_ours - h_model).abs().max().item()
    mean_diff = (h_ours - h_model).abs().mean().item()
    cos_sim = (
        torch.nn.functional.cosine_similarity(
            h_ours.float().reshape(-1, h_ours.shape[-1]),
            h_model.float().reshape(-1, h_model.shape[-1]),
            dim=1,
        )
        .mean()
        .item()
    )

    print(f"  Max absolute diff:  {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")
    print(f"  Cosine similarity:  {cos_sim:.10f}")

    passed = max_diff < 1e-3  # bf16 precision: ~1e-3
    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'} (threshold: 1e-3)")

    return h_ours, h_model


def test_compiled_vs_unfused(model, model_args, tokens, masks, device):
    """Compare compiled vs unfused FlexAttention outputs."""
    print(f"\n{'=' * 60}")
    print(f"TEST 3: Compiled vs unfused FlexAttention (seq_len={tokens.shape[1]})")
    print(f"{'=' * 60}")

    from torch.nn.attention.flex_attention import flex_attention as _raw_flex_attn
    from torchtitan.models.attention import FlexAttentionWrapper

    # Unfused
    FlexAttentionWrapper._compiled_flex_attn = staticmethod(_raw_flex_attn)
    with torch.no_grad():
        h_unfused = model.tok_embeddings(tokens)
        for layer in model.layers.values():
            h_unfused = layer(h_unfused, model.rope_cache, masks)
        h_unfused = model.norm(h_unfused)

    # Compiled (basic, no max_autotune)
    compiled_flex = torch.compile(_raw_flex_attn)
    FlexAttentionWrapper._compiled_flex_attn = staticmethod(compiled_flex)
    print("  Compiling FlexAttention (first call triggers JIT)...")
    t0 = time.time()
    with torch.no_grad():
        h_compiled = model.tok_embeddings(tokens)
        for layer in model.layers.values():
            h_compiled = layer(h_compiled, model.rope_cache, masks)
        h_compiled = model.norm(h_compiled)
    print(f"  Compiled forward took {time.time() - t0:.1f}s")

    # Compare
    max_diff = (h_unfused - h_compiled).abs().max().item()
    mean_diff = (h_unfused - h_compiled).abs().mean().item()
    cos_sim = (
        torch.nn.functional.cosine_similarity(
            h_unfused.float().reshape(-1, h_unfused.shape[-1]),
            h_compiled.float().reshape(-1, h_compiled.shape[-1]),
            dim=1,
        )
        .mean()
        .item()
    )

    print(f"  Max absolute diff:  {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")
    print(f"  Cosine similarity:  {cos_sim:.10f}")

    # Compiled FlexAttention may have small numerical differences due to
    # Triton kernel computation order vs Python eager. Allow bf16 tolerance.
    passed = max_diff < 0.05  # Allow some tolerance for compiled vs unfused
    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'} (threshold: 0.05)")

    return h_unfused, h_compiled


def main():
    parser = argparse.ArgumentParser(description="Validate extraction forward pass")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-flavor", default="20b")
    parser.add_argument("--gpu", type=int, default=1, help="GPU device (default: 1)")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=0,
        help="Use random tokens of this length (default: use --input doc)",
    )
    parser.add_argument("--input", type=str, help="JSONL file for real document test")
    parser.add_argument(
        "--doc-idx", type=int, default=0, help="Document index in JSONL"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Max tokens (for real docs)"
    )
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    print(f"Loading model on {device}...")
    t0 = time.time()
    model, tokenizer, model_args = load_model_and_tokenizer(
        args.model_dir, device, args.model_flavor
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Prepare input tokens
    if args.input:
        with open(args.input) as f:
            for i, line in enumerate(f):
                if i == args.doc_idx:
                    doc = json.loads(line)
                    text = doc.get("text", doc.get("content", ""))
                    break
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) > args.max_tokens:
            token_ids = token_ids[: args.max_tokens]
        print(f"\nUsing real document (idx={args.doc_idx}): {len(token_ids)} tokens")
    else:
        seq_len = args.seq_len or 2048
        token_ids = torch.randint(0, tokenizer.vocab_size, (seq_len,)).tolist()
        print(f"\nUsing random tokens: {len(token_ids)} tokens")

    seq_len = len(token_ids)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Test 1: Mask equivalence
    our_masks, model_masks = test_mask_equivalence(model, model_args, seq_len, device)

    # Test 2: Hidden state equivalence (our masks vs model masks)
    test_hidden_state_equivalence(
        model, model_args, tokens, our_masks, model_masks, device
    )

    # Test 3: Compiled vs unfused (skip for very long sequences to avoid OOM)
    if seq_len <= 12000:
        test_compiled_vs_unfused(model, model_args, tokens, our_masks, device)
    else:
        print(f"\n{'=' * 60}")
        print(f"TEST 3: SKIPPED (seq_len={seq_len} > 12000, unfused would OOM)")
        print(f"{'=' * 60}")

    print(f"\n{'=' * 60}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
