#!/usr/bin/env python3
"""
Quick validation: compiled vs unfused FlexAttention — check mean-pooled
embedding similarity (what we actually use for clustering).
"""

import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/mnt/git/torchtitan.temp")


def main():
    device = "cuda:1"
    torch.cuda.set_device(1)

    from transformers import AutoTokenizer
    from torchtitan.models.gpt_oss import gptoss_configs
    from torchtitan.models.gpt_oss.model.model import GptOssModel, precompute_rope_cache
    from torchtitan.models.gpt_oss.model.state_dict_adapter import (
        GptOssStateDictAdapter,
    )

    model_dir = "/mnt/models/openai/gpt-oss-20b"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_args = gptoss_configs["20b"]

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
    print(f"Model loaded: {torch.cuda.memory_allocated(device) / 1024**3:.1f} GiB")

    from torch.nn.attention.flex_attention import (
        flex_attention as _raw_flex_attn,
        and_masks,
        create_block_mask,
    )
    from torchtitan.models.attention import (
        FlexAttentionWrapper,
        get_causal_mask_mod,
        get_sliding_window_mask_mod,
    )

    # Load a real document
    text = None
    with open("/mnt/models/persona_datasets/persona-iota-v2.jsonl") as f:
        for i, line in enumerate(f):
            if i == 100:
                text = json.loads(line)["text"]
                break
    if text is None:
        print("ERROR: Input file has fewer than 101 lines")
        sys.exit(1)

    for seq_len in [1024, 2048, 4096, 8192]:
        token_ids = tokenizer.encode(text, add_special_tokens=False)[:seq_len]
        actual_len = len(token_ids)
        tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

        # Create masks
        causal = get_causal_mask_mod()
        sliding = get_sliding_window_mask_mod(model_args.sliding_window_size)
        masks = {
            "basic_mask": create_block_mask(
                and_masks(causal), 1, None, actual_len, actual_len
            ),
            "sliding_window_mask": create_block_mask(
                and_masks(causal, sliding), 1, None, actual_len, actual_len
            ),
        }

        def forward_and_pool(flex_attn_fn):
            FlexAttentionWrapper._compiled_flex_attn = staticmethod(flex_attn_fn)
            with torch.no_grad():
                h = model.tok_embeddings(tokens)
                for layer in model.layers.values():
                    h = layer(h, model.rope_cache, masks)
                h = model.norm(h)
            # Mean pool
            h_f32 = h.float()
            mean_emb = h_f32.mean(dim=1)  # [1, dim]
            return h, F.normalize(mean_emb, dim=1)

        # Unfused
        h_unfused, emb_unfused = forward_and_pool(_raw_flex_attn)
        torch.cuda.empty_cache()

        # Compiled
        compiled_flex = torch.compile(_raw_flex_attn)
        h_compiled, emb_compiled = forward_and_pool(compiled_flex)

        # Per-position analysis
        pos_cos = F.cosine_similarity(
            h_unfused.float().squeeze(0),
            h_compiled.float().squeeze(0),
            dim=1,
        )
        pos_max_diff = (h_unfused - h_compiled).abs().max().item()
        pos_mean_diff = (h_unfused - h_compiled).abs().mean().item()

        # Mean-pooled embedding comparison (what we use for clustering)
        emb_cos = F.cosine_similarity(emb_unfused, emb_compiled, dim=1).item()
        emb_l2 = (emb_unfused - emb_compiled).norm().item()

        print(f"\nseq_len={actual_len}")
        print(
            f"  Per-position: max_diff={pos_max_diff:.2f}, mean_diff={pos_mean_diff:.4f}, "
            f"cos_sim={pos_cos.mean():.6f} (min={pos_cos.min():.4f})"
        )
        print(f"  Mean-pooled:  cos_sim={emb_cos:.8f}, L2_dist={emb_l2:.6f}")

        # Clean up for next iteration
        del h_unfused, h_compiled, emb_unfused, emb_compiled, masks
        torch.cuda.empty_cache()
        # Reset compiled flex to avoid stale graphs
        torch._dynamo.reset()


if __name__ == "__main__":
    main()
