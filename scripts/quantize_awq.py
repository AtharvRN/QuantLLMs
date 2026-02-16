#!/usr/bin/env python3
"""Quantize a HF model to AWQ (4-bit) using AutoAWQ.

Example:
  python scripts/quantize_awq.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --out models/llama3.1-8b-instruct-awq
"""
from __future__ import annotations

import argparse
import inspect
from typing import Iterable, List

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantize a model with AWQ")
    p.add_argument("--model", required=True, help="Base HF model id or path")
    p.add_argument("--out", required=True, help="Output directory for quantized model")
    p.add_argument("--w-bit", type=int, default=4, help="Weight bits")
    p.add_argument("--group-size", type=int, default=128, help="Group size")
    p.add_argument("--zero-point", action="store_true", default=True, help="Use zero-point quantization")
    p.add_argument("--no-zero-point", dest="zero_point", action="store_false")
    p.add_argument("--version", default="GEMM", help="AWQ kernel version (GEMM/GEMV)")
    p.add_argument("--calib-dataset", default="wikitext2", help="Calibration dataset (wikitext2)")
    p.add_argument("--calib-samples", type=int, default=128, help="Calibration samples")
    p.add_argument("--max-seq-len", type=int, default=512, help="Max tokens per sample")
    return p.parse_args()


def load_calib_texts(name: str, n: int) -> List[str]:
    if name != "wikitext2":
        raise ValueError("Only wikitext2 is supported for now")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts: List[str] = []
    for row in ds:
        text = row.get("text", "").strip()
        if not text:
            continue
        texts.append(text)
        if len(texts) >= n:
            break
    return texts


def main() -> int:
    args = parse_args()
    try:
        from awq import AutoAWQForCausalLM
    except Exception as exc:
        raise SystemExit("autoawq is required: pip install autoawq") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoAWQForCausalLM.from_pretrained(args.model)

    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.group_size,
        "w_bit": args.w_bit,
        "version": args.version,
    }

    calib_texts = load_calib_texts(args.calib_dataset, args.calib_samples)

    # Use calib_data if supported by installed autoawq
    sig = inspect.signature(model.quantize)
    if "calib_data" in sig.parameters:
        model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)
    else:
        model.quantize(tokenizer, quant_config=quant_config)

    model.save_quantized(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Wrote AWQ model to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
