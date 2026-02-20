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

## AdvBench (refusal & jailbreak stress)
- Prompts: generate with `datasets` → `data/advbench/advbench.jsonl`.
- Run FP vs quantized (GGUF) with vLLM to produce
  - `output/advbench_qwen3_fp.jsonl`
  - `output/advbench_qwen3_gguf.jsonl`
- Score refusals/flips: `python scripts/score_advbench_outputs.py --a output/advbench_qwen3_fp.jsonl --b output/advbench_qwen3_gguf.jsonl`.
- Black-box attacks (prompt-space): `scripts/run_advbench_attacks.py` with `--attack shallow|uja|rephrase` to measure ASR under jailbreaks.

## Safety neuron discovery (GTAC-style)
- Script: `scripts/find_safety_neurons.py`
- Inputs: safe and unsafe prompt JSONL (`prompt` field). Computes per-neuron activation change scores (safe vs unsafe) over MLP layers, last-token activations.
- Example:
```
python scripts/find_safety_neurons.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --safe data/safe_prompts.jsonl \
  --unsafe data/advbench/advbench.jsonl \
  --max-prompts 200 \
  --topk-fraction 0.05 \
  --out output/safety_neurons_qwen3.json
```
- Output: ranked list of neuron indices/scores (first-stage identification; no causal patching).

## Safety neuron causal patch demo (Transformers-only)
- Script: `scripts/patch_safety_neurons_demo.py`
- Uses top neurons from `find_safety_neurons.py`, caches safe activations on a benign prompt, and patches those neurons during generation on harmful prompts to see refusal flips.
- Example:
```
python scripts/patch_safety_neurons_demo.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --top-neurons output/safety_neurons_qwen3_beavertails.json \
  --harmful data/advbench/advbench.jsonl \
  --benign data/beavertails/benign.jsonl \
  --num-harmful 10 \
  --num-top 200
```
- Prints refusal rates before/after patching on the sampled harmful prompts. (Demo-scale; not vLLM.)

## Outputs
1. JSONL with model generations and extracted answers.
2. JSON mapping of `id -> predicted_index` for submission/scoring.
