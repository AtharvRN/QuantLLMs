#!/usr/bin/env python3
"""Download SafetyBench via datasets and write JSONL for offline runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download SafetyBench split to JSONL")
    p.add_argument(
        "--split",
        default="test_en",
        choices=["test_en", "test_zh", "test_zh_subset", "dev_en", "dev_zh"],
        help="SafetyBench split",
    )
    p.add_argument(
        "--out",
        default="data/safetybench/test_en.jsonl",
        help="Output JSONL path",
    )
    return p.parse_args()


def load_safetybench(split: str):
    if split.startswith("dev_"):
        cfg = "dev"
        lang = split.split("_", 1)[1]
        return load_dataset("thu-coai/SafetyBench", cfg)[lang]

    if split == "test_zh_subset":
        try:
            return load_dataset("thu-coai/SafetyBench", "test_zh_subset")["zh"]
        except Exception:
            pass

    cfg = "test"
    lang = split.split("_", 1)[1]
    return load_dataset("thu-coai/SafetyBench", cfg)[lang]


def main() -> int:
    args = parse_args()
    dataset = load_safetybench(args.split)
    rows: List[Dict[str, Any]] = [dict(r) for r in dataset]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
