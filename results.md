# Results Snapshot

## SafetyBench (test_en)
- Datasets: `data/safetybench/test_en.jsonl` (zero-shot).
- Models run so far: Qwen3-4B FP (`output/safetybench/qwen3-4b-instruct-2507-fp.preds.json`) and Qwen3-4B GGUF (`output/safetybench/qwen3-4b-gguf.preds.json`).
- Comparison files: `output/safetybench/compare_qwen3_fp_vs_gguf.json`, `output/safetybench/compare_qwen3_instruct2507_fp_vs_gguf.json` (see `docs/SAFETYBENCH.md` for metric details).
- Observation: FP and GGUF are effectively tied on SafetyBench accuracy; no safety degradation detected.

## AdvBench (refusal stress test)
Prompts regenerated from `AlignmentResearch/AdvBench` → `data/advbench/advbench.jsonl` (520 harmful prompts).

Runs (vLLM, Qwen3-4B):
- FP model: `Qwen/Qwen3-4B-Instruct-2507` → `output/advbench_qwen3_fp.jsonl`
- Quant model: `models/qwen3-4b-gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` → `output/advbench_qwen3_gguf.jsonl`

Scoring (refusal heuristic in `scripts/score_advbench_outputs.py`):
- FP: refusals 514 / 520 (98.85%), compliance 1.15%, avg chars ≈ 356
- GGUF: refusals 516 / 520 (99.23%), compliance 0.77%, avg chars ≈ 484
- Flip rate FP↔GGUF: 0.77% (1 FP→comply / GGUF→refuse, 3 FP→refuse / GGUF→comply)
- Takeaway: Quantized behaves nearly identical; slightly more conservative.

## New black-box attack harness
`scripts/run_advbench_attacks.py` adds prompt-space attacks:
- `--attack shallow`: ShallowJail-style system prompt prepend.
- `--attack uja`: iterative prompt-suffix search (query-only), `--attempts N` (default 6).

Examples:
```
# Shallow on FP
python scripts/run_advbench_attacks.py \
  --data data/advbench/advbench.jsonl \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --attack shallow \
  --out output/advbench_qwen3_fp_shallow.jsonl

# UJA on GGUF
python scripts/run_advbench_attacks.py \
  --data data/advbench/advbench.jsonl \
  --model models/qwen3-4b-gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --attack uja \
  --attempts 8 \
  --out output/advbench_qwen3_gguf_uja.jsonl
```

Scoring attacks:
```
python scripts/score_advbench_outputs.py --a output/advbench_qwen3_fp_shallow.jsonl --b output/advbench_qwen3_gguf_shallow.jsonl
```

Recommended next: run shallow + UJA on both FP and GGUF, then compute ASR (1 - refusal rate) and flip analysis.
