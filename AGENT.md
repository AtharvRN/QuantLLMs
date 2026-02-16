# QuantLLMs Agent Notes

This repo contains scripts to run safety-focused LLM evaluations using vLLM. The primary workflow added here is a SafetyBench zero-shot evaluation against `test_en`.

## Quickstart
1. Install dependencies.
2. Run the SafetyBench script with a vLLM-compatible model.
3. Inspect the JSONL outputs and optional accuracy metrics.

## Dependencies
1. Python deps from `requirements.txt`.
2. `vllm` installed separately if not already available.

## SafetyBench Zero-Shot (test_en)
1. Run:
```
python /Users/atharvramesh/UCSD/Winter2026/ECE285/QuantLLMs/scripts/run_safetybench_vllm.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --split test_en \
  --out /Users/atharvramesh/UCSD/Winter2026/ECE285/QuantLLMs/output/safetybench/test_en.jsonl \
  --pred-json /Users/atharvramesh/UCSD/Winter2026/ECE285/QuantLLMs/output/safetybench/test_en.preds.json
```
2. Optional: pass `--label-path` to compute accuracy if you have official labels.
3. See `docs/SAFETYBENCH.md` for details and troubleshooting.

## Outputs
1. JSONL with model generations and extracted answers.
2. JSON mapping of `id -> predicted_index` for submission/scoring.
