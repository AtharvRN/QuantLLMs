# Experiment Log

## Environment
- Repo: /workspace/QuantLLMs (pod: atharv-rwx-pod)
- Main models used: Qwen/Qwen3-4B-Instruct-2507 (FP); GGUF quant: models/qwen3-4b-gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf (vLLM only)
- Key datasets prepared: SafetyBench test_en; AdvBench (520 harmful prompts); BeaverTails harmful/benign (2,000 each) at data/beavertails/{harmful,benign}.jsonl

## SafetyBench
- Runs (vLLM):
  - FP preds: output/safetybench/qwen3-4b-instruct-2507-fp.preds.json
  - GGUF preds: output/safetybench/qwen3-4b-gguf.preds.json
- Comparisons: output/safetybench/compare_qwen3_fp_vs_gguf.json and output/safetybench/compare_qwen3_instruct2507_fp_vs_gguf.json
- Result: FP and GGUF essentially tied; no noticeable safety accuracy degradation.

## AdvBench (baseline refusal)
- Prompts: data/advbench/advbench.jsonl (520 harmful)
- Runs (vLLM):
  - FP: output/advbench_qwen3_fp.jsonl
  - GGUF: output/advbench_qwen3_gguf.jsonl
- Refusal stats (score_advbench_outputs.py):
  - FP: refusals 514 / 520 (98.85%), compliance 1.15%, avg chars ≈ 356
  - GGUF: refusals 516 / 520 (99.23%), compliance 0.77%, avg chars ≈ 484
  - Flip rate FP↔GGUF: 0.77% (1 FP→comply, 3 GGUF→comply)

## AdvBench attacks (prompt-space)
- Harness: scripts/run_advbench_attacks.py
- Shallow attack results:
  - FP: output/advbench_qwen3_fp_shallow.jsonl, refusals 517 / 520 (99.42%), compliance 0.58%, avg chars ≈ 306
  - GGUF: output/advbench_qwen3_gguf_shallow.jsonl, refusals 513 / 520 (98.65%), compliance 1.35%, avg chars ≈ 629
  - Flips: 5 FP→refuse & GGUF→comply; 1 FP→comply & GGUF→refuse (flip rate 1.15%)
- UJA and rephrase modes added but not yet run (no results recorded).

## Safety neurons (identification)
- Script: scripts/find_safety_neurons.py
- Data: safe= data/beavertails/benign.jsonl (500 sampled); unsafe= data/beavertails/harmful.jsonl (500 sampled)
- Model: Qwen/Qwen3-4B-Instruct-2507 (FP)
- Output: output/safety_neurons_qwen3_beavertails.json (61,747 ranked neurons; top entries around layers 34–35)
- Quantized (AWQ/GGUF) locator not run (Transformers cannot load GGUF; AWQ requires autoawq not installed).

## Safety neuron causal patch demo
- Script: scripts/patch_safety_neurons_demo.py
- Settings: top-neurons= output/safety_neurons_qwen3_beavertails.json, benign cache= data/beavertails/benign.jsonl, harmful= data/advbench/advbench.jsonl
- Runs:
  - 10 harmful prompts, top 200 neurons: base refusal 1.0, patched 1.0 (no change)
  - 400 harmful prompts, top 200 neurons: base refusal 0.9425, patched 0.9425 (no change)
- Note: likely ceilinged prompts; stronger tests pending (harder prompts, larger top-k, richer cache).

## Data prep
- BeaverTails splits created: data/beavertails/harmful.jsonl (2,000) and data/beavertails/benign.jsonl (2,000).

## Pending / next
- Run attacks (UJA/rephrase) and HarmBench classifier on FP vs GGUF outputs.
- Compare safety neurons across model variants (needs Transformers-loadable quant or fine-tuned FP).
- Try causal patching on “hard” prompts (non-refusals) with larger top-k (e.g., 1,000) and richer benign cache.
