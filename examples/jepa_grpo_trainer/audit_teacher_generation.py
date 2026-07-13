#!/usr/bin/env python3
"""Measure full-thinking Qwen teacher response lengths before cache generation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch

from verl.experimental.fepo.math_parser import compute_math_reward


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", required=True)
    p.add_argument("--teacher-model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--seed", type=int, default=31415)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--num-shards", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=32768)
    p.add_argument("--gpu-mem-frac", type=float, default=0.92)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    df = pd.read_parquet(args.train_file)
    n = min(args.n_prompts, len(df))
    # Both shard processes repeat this deterministic sample, then take disjoint
    # contiguous slices from it.
    sampled = df.sample(n=n, random_state=args.seed)
    start = n * args.shard_index // args.num_shards
    end = n * (args.shard_index + 1) // args.num_shards
    rows = sampled.iloc[start:end].to_dict("records")
    print(f"[audit] shard {args.shard_index + 1}/{args.num_shards}: {len(rows)} prompts", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    prompts = [
        tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        for r in rows
    ]
    llm = LLM(
        model=args.teacher_model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem_frac,
        trust_remote_code=True,
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(n=1, temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.max_new_tokens),
    )
    results = []
    for row, output in zip(rows, outputs):
        text = output.outputs[0].text
        result = compute_math_reward(
            text, row["reward_model"]["ground_truth"], dataset_kind=row.get("data_source")
        )
        results.append({
            "index": int(row["extra_info"]["index"]),
            "n_tokens": len(output.outputs[0].token_ids),
            "is_correct": bool(result.is_correct),
            "finish_reason": output.outputs[0].finish_reason,
        })
    lengths = torch.tensor([r["n_tokens"] for r in results], dtype=torch.float64)
    summary = {
        "n": len(results),
        "mean_tokens": float(lengths.mean()),
        "median_tokens": float(lengths.median()),
        "p90_tokens": float(torch.quantile(lengths, 0.90)),
        "max_tokens": int(lengths.max()),
        "correct": int(sum(r["is_correct"] for r in results)),
        "capped": int(sum(r["finish_reason"] == "length" for r in results)),
    }
    out = Path(args.out)
    tmp = out.with_name(out.name + ".tmp")
    torch.save({"summary": summary, "samples": results}, tmp)
    os.replace(tmp, out)
    print(f"[audit] {summary}", flush=True)


if __name__ == "__main__":
    main()
