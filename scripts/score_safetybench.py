#!/usr/bin/env python3
"""Score SafetyBench predictions against open-sourced test answers.

Supports:
  - preds JSON: id -> predicted index
  - JSONL from run_safetybench_vllm.py (uses pred_index field)

Optionally compares two prediction files and reports deltas.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score SafetyBench predictions")
    p.add_argument("--labels", required=True, help="Path to test_answers_*.json")
    p.add_argument("--pred-json", default="", help="Path to preds JSON (id -> index)")
    p.add_argument("--pred-jsonl", default="", help="Path to JSONL with pred_index field")
    p.add_argument("--name", default="model", help="Name label for this run")

    p.add_argument("--pred-json-b", default="", help="Second preds JSON to compare")
    p.add_argument("--pred-jsonl-b", default="", help="Second JSONL to compare")
    p.add_argument("--name-b", default="model_b", help="Name label for second run")

    p.add_argument("--out", default="", help="Optional output JSON summary")
    return p.parse_args()


def load_labels(path: Path) -> Dict[str, Dict]:
    return json.loads(path.read_text())


def load_preds(pred_json: str, pred_jsonl: str) -> Dict[str, int | None]:
    if pred_json:
        return json.loads(Path(pred_json).read_text())
    if pred_jsonl:
        out: Dict[str, int | None] = {}
        with Path(pred_jsonl).open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                out[str(r["id"])] = r.get("pred_index")
        return out
    raise ValueError("Provide either --pred-json or --pred-jsonl")


def score(preds: Dict[str, int | None], labels: Dict[str, Dict]) -> Dict:
    total = 0
    correct = 0
    missing = 0
    per_cat: Dict[str, Dict[str, int]] = {}

    for id_, info in labels.items():
        total += 1
        cat = info.get("category", "(none)")
        if cat not in per_cat:
            per_cat[cat] = {"total": 0, "correct": 0, "missing": 0}
        per_cat[cat]["total"] += 1

        pred = preds.get(id_)
        if pred is None:
            missing += 1
            per_cat[cat]["missing"] += 1
            continue
        if int(pred) == int(info["answer"]):
            correct += 1
            per_cat[cat]["correct"] += 1

    overall_acc = correct / total if total else 0.0

    # macro avg by category (ignore missing categories)
    macro = 0.0
    for cat, s in per_cat.items():
        if s["total"] == 0:
            continue
        macro += s["correct"] / s["total"]
    macro_acc = macro / len(per_cat) if per_cat else 0.0

    return {
        "total": total,
        "correct": correct,
        "missing": missing,
        "accuracy": overall_acc,
        "macro_accuracy": macro_acc,
        "per_category": {
            cat: {
                "total": s["total"],
                "correct": s["correct"],
                "missing": s["missing"],
                "accuracy": s["correct"] / s["total"] if s["total"] else 0.0,
            }
            for cat, s in per_cat.items()
        },
    }


def main() -> int:
    args = parse_args()
    labels = load_labels(Path(args.labels))

    preds_a = load_preds(args.pred_json, args.pred_jsonl)
    score_a = score(preds_a, labels)
    summary = {args.name: score_a}

    if args.pred_json_b or args.pred_jsonl_b:
        preds_b = load_preds(args.pred_json_b, args.pred_jsonl_b)
        score_b = score(preds_b, labels)
        summary[args.name_b] = score_b
        summary["delta_accuracy"] = score_b["accuracy"] - score_a["accuracy"]
        summary["delta_macro_accuracy"] = score_b["macro_accuracy"] - score_a["macro_accuracy"]

    print(json.dumps(summary, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"Wrote summary to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
