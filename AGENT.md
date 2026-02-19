# QuantLLMs Agent Notes

This repo contains scripts to run safety-focused LLM evaluations using vLLM. Primary workflows: SafetyBench (zero-shot, `test_en`) and AdvBench refusal-rate stress tests.

## Quickstart
1. Install dependencies.
2. Run the SafetyBench script with a vLLM-compatible model.
3. Inspect the JSONL outputs and optional accuracy metrics.

## Dependencies
1. Python deps from `requirements.txt`.
2. `vllm` installed separately if not already available.

## SafetyBench Zero-Shot (test_en)
1. Download the split (offline):
```
python scripts/download_safetybench.py \
  --split test_en \
  --out data/safetybench/test_en.jsonl
```
2. Run:
```
python scripts/run_safetybench_vllm.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --split test_en \
  --input-jsonl data/safetybench/test_en.jsonl \
  --out output/safetybench/test_en.jsonl \
  --pred-json output/safetybench/test_en.preds.json
```
3. Optional: pass `--label-path` to compute accuracy if you have official labels.
4. See `docs/SAFETYBENCH.md` for details and troubleshooting.

## AdvBench (refusal stress test)
- Prompts: generate with `datasets` and save to `data/advbench/advbench.jsonl` (script snippet in `results.md`).
- Run FP vs quantized models (vLLM) using the inline loop pattern in `results.md` to create
  - `output/advbench_qwen3_fp.jsonl`
  - `output/advbench_qwen3_gguf.jsonl`
- Score refusals and flips: `python scripts/score_advbench_outputs.py --a output/advbench_qwen3_fp.jsonl --b output/advbench_qwen3_gguf.jsonl`.

## Outputs
1. JSONL with model generations and extracted answers.
2. JSON mapping of `id -> predicted_index` for submission/scoring.
