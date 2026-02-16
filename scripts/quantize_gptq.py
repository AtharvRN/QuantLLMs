#!/usr/bin/env python3
"""Quantize a HF model to GPTQ (4-bit) using AutoGPTQ.

Example:
  python scripts/quantize_gptq.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --out models/llama3.1-8b-instruct-gptq
"""
from __future__ import annotations

import argparse
from typing import List

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantize a model with GPTQ")
    p.add_argument("--model", required=True, help="Base HF model id or path")
    p.add_argument("--out", required=True, help="Output directory for quantized model")
    p.add_argument("--bits", type=int, default=4, help="Quantization bits")
    p.add_argument("--group-size", type=int, default=128, help="Group size")
    p.add_argument("--desc-act", action="store_true", default=False, help="Use activation order")
    p.add_argument("--damp-percent", type=float, default=0.01, help="Dampening factor")
    p.add_argument("--calib-dataset", default="wikitext2", help="Calibration dataset (wikitext2)")
    p.add_argument("--calib-samples", type=int, default=128, help="Calibration samples")
    p.add_argument("--max-seq-len", type=int, default=512, help="Max tokens per sample")
    p.add_argument("--device", default="cuda:0", help="Quantization device")
    return p.parse_args()


def load_calib_tokens(model: str, name: str, n: int, max_len: int) -> List[List[int]]:
    if name != "wikitext2":
        raise ValueError("Only wikitext2 is supported for now")
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokens: List[List[int]] = []
    for row in ds:
        text = row.get("text", "").strip()
        if not text:
            continue
        ids = tokenizer(text, truncation=True, max_length=max_len).get("input_ids", [])
        if not ids:
            continue
        tokens.append(ids)
        if len(tokens) >= n:
            break
    return tokens


def main() -> int:
    args = parse_args()
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except Exception as exc:
        raise SystemExit("auto-gptq is required: pip install auto-gptq") from exc

    quant_config = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
        damp_percent=args.damp_percent,
    )

    model = AutoGPTQForCausalLM.from_pretrained(
        args.model,
        quantize_config=quant_config,
        device=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    calib_data = load_calib_tokens(args.model, args.calib_dataset, args.calib_samples, args.max_seq_len)
    model.quantize(calib_data)

    model.save_quantized(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Wrote GPTQ model to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
