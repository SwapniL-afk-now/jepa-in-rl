#!/usr/bin/env python3
"""Merge disjoint verified-teacher text-cache shards into one training cache."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("inputs", nargs="+", help="Shard .responses.pt files")
    args = parser.parse_args()

    merged: dict[int, list[str]] = {}
    for filename in args.inputs:
        shard = torch.load(filename, map_location="cpu", weights_only=False)
        for raw_index, responses in shard.items():
            index = int(raw_index)
            if index in merged:
                raise ValueError(f"overlapping dataset index {index} in {filename}")
            values = list(responses)
            if not values or not all(isinstance(x, str) for x in values):
                raise ValueError(f"invalid response list at index {index} in {filename}")
            merged[index] = values

    out = Path(args.out)
    tmp = out.with_name(out.name + ".tmp")
    torch.save(dict(sorted(merged.items())), tmp)
    os.replace(tmp, out)
    print(f"merged {len(args.inputs)} shards: {len(merged)} questions, "
          f"{sum(map(len, merged.values()))} verified responses -> {out}")


if __name__ == "__main__":
    main()
