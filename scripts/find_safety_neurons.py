#!/usr/bin/env python3
"""Minimal safety-neuron locator (change-score only).

This is a simplified re-implementation of the "Generation-Time Activation
Contrasting" stage from SafetyNeuron (arXiv:2406.14144), designed to be easy to
run on a single GPU with HF Transformers models.

What it does
------------
1) Loads a model and tokenizer (HF `model_id`).
2) Reads two prompt sets: SAFE and UNSAFE (JSONL with field `prompt`).
3) Runs a forward pass on each prompt (no generation) and collects hidden MLP
   activations (last token position).
4) Computes a per-neuron change score = |mean_safe - mean_unsafe|.
5) Saves ranked neuron indices with scores to JSON.

Limitations
-----------
- Uses the last-token activation instead of full decoding-time traces
  (good enough to triage candidates; lighter than full generation hooks).
- Only supports decoder-only models where MLP modules are accessible.
- Does NOT perform causal patching; this is the identification step only.

Usage
-----
python scripts/find_safety_neurons.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --safe data/safe_prompts.jsonl \
  --unsafe data/unsafe_prompts.jsonl \
  --max-prompts 200 \
  --topk-fraction 0.05 \
  --out safety_neurons_qwen25.json

Outputs
-------
JSON with fields:
  {
    "layer_neurons": [ {"layer": i, "idx": j, "score": s}, ... ranked ... ],
    "meta": {model, safe_count, unsafe_count}
  }

Author: Codex helper (derived conceptually from SafetyNeuron repo).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class HookSpec:
    layer_id: int
    name: str
    module: torch.nn.Module


def read_prompts(path: str, limit: int | None) -> List[str]:
    prompts = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            p = obj.get("prompt", "").strip()
            if not p:
                continue
            prompts.append(p)
            if limit and len(prompts) >= limit:
                break
    return prompts


def find_mlp_modules(model) -> List[HookSpec]:
    specs = []
    for name, module in model.named_modules():
        # Heuristic: capture MLP/FFN output before residual add.
        lname = name.lower()
        if any(tag in lname for tag in ["mlp", "ffn", "feed_forward", "ffn_up"]):
            # Identify layer id if present in the path
            layer_id = -1
            for part in lname.split("."):
                if part.isdigit():
                    layer_id = int(part)
                    break
            specs.append(HookSpec(layer_id=layer_id, name=name, module=module))
    # Keep only unique modules at the lowest depth (avoid duplicates)
    # Sort by layer_id then name for stable ordering
    specs = sorted(specs, key=lambda x: (x.layer_id, x.name))
    return specs


def collect_activations(model, tokenizer, prompts: List[str], specs: List[HookSpec], device: str) -> List[torch.Tensor]:
    """Return list of tensors [len(specs)] each shaped (total_samples, hidden)."""
    buffers = [ [] for _ in specs ]

    hooks = []
    for i, spec in enumerate(specs):
        def make_hook(idx):
            def hook(_, __, output):
                # output shape: (batch, seq, hidden) or (batch, hidden)
                if output.dim() == 3:
                    buf = output[:, -1, :].detach()  # last token
                else:
                    buf = output.detach()
                buffers[idx].append(buf.cpu())
            return hook
        hooks.append(spec.module.register_forward_hook(make_hook(i)))

    model.eval()
    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            model(**inputs)

    for h in hooks:
        h.remove()

    activations = []
    for buf in buffers:
        if not buf:
            activations.append(torch.empty(0))
        else:
            activations.append(torch.cat(buf, dim=0))  # (N, hidden)
    return activations


def compute_change_scores(safe_acts: List[torch.Tensor], unsafe_acts: List[torch.Tensor]) -> List[Tuple[int, int, float]]:
    """Return list of (layer_id, neuron_idx, score)."""
    results = []
    for acts_safe, acts_unsafe in zip(safe_acts, unsafe_acts):
        if acts_safe.numel() == 0 or acts_unsafe.numel() == 0:
            continue
        mean_safe = acts_safe.mean(dim=0)
        mean_unsafe = acts_unsafe.mean(dim=0)
        scores = (mean_safe - mean_unsafe).abs()
        for j, s in enumerate(scores.tolist()):
            results.append((acts_safe.shape[-1], j, s))
    return results


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--tokenizer", default=None, help="Optional tokenizer override")
    ap.add_argument("--safe", required=True, help="JSONL with field 'prompt'")
    ap.add_argument("--unsafe", required=True, help="JSONL with field 'prompt'")
    ap.add_argument("--max-prompts", type=int, default=0, help="Limit per split (0 = all)")
    ap.add_argument("--topk-fraction", type=float, default=0.05, help="Fraction of neurons to keep")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="Output JSON path")
    return ap.parse_args()


def main():
    args = parse_args()
    safe_prompts = read_prompts(args.safe, args.max_prompts or None)
    unsafe_prompts = read_prompts(args.unsafe, args.max_prompts or None)
    print(f"Loaded {len(safe_prompts)} safe, {len(unsafe_prompts)} unsafe prompts")

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=args.device)

    specs = find_mlp_modules(model)
    if not specs:
        raise ValueError("Could not find MLP/FFN modules to hook")
    print(f"Hooking {len(specs)} MLP modules")

    safe_acts = collect_activations(model, tok, safe_prompts, specs, args.device)
    unsafe_acts = collect_activations(model, tok, unsafe_prompts, specs, args.device)

    # compute change scores layer-wise
    all_scores = []
    for spec, s_act, u_act in zip(specs, safe_acts, unsafe_acts):
        if s_act.numel() == 0 or u_act.numel() == 0:
            continue
        mean_safe = s_act.mean(dim=0)
        mean_unsafe = u_act.mean(dim=0)
        scores = (mean_safe - mean_unsafe).abs()
        for j, score in enumerate(scores.tolist()):
            all_scores.append({"layer": spec.layer_id, "name": spec.name, "idx": j, "score": score})

    all_scores = sorted(all_scores, key=lambda x: x["score"], reverse=True)
    k = max(1, int(len(all_scores) * args.topk_fraction))
    top_scores = all_scores[:k]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "meta": {
                "model": args.model,
                "safe_count": len(safe_prompts),
                "unsafe_count": len(unsafe_prompts),
                "topk_fraction": args.topk_fraction,
            },
            "layer_neurons": top_scores,
        }, f, indent=2)

    print(f"Wrote {len(top_scores)} neurons to {out_path}")


if __name__ == "__main__":
    main()

