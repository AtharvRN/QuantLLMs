# SafetyBench Evaluation Setup

This document describes how to run SafetyBench using the vLLM `LLM()` API with a self-contained script.

## What This Script Does
1. Downloads SafetyBench via `datasets`.
2. Formats each example as a multiple-choice question with options A-D.
3. Runs vLLM generation in batches.
4. Extracts a single-letter prediction.
5. Writes a JSONL of generations and a JSON mapping `id -> predicted_index`.

## Install
1. `pip install -r /Users/atharvramesh/UCSD/Winter2026/ECE285/QuantLLMs/requirements.txt`
2. `pip install vllm`

## Step 1: Download Data (offline run)
```
python /Users/atharvramesh/UCSD/Winter2026/ECE285/QuantLLMs/scripts/download_safetybench.py \
  --split test_en \
  --out /Users/atharvramesh/UCSD/Winter2026/ECE285/QuantLLMs/data/safetybench/test_en.jsonl
```

## Zero-Shot test_en
```
python /Users/atharvramesh/UCSD/Winter2026/ECE285/QuantLLMs/scripts/run_safetybench_vllm.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --split test_en \
  --input-jsonl /Users/atharvramesh/UCSD/Winter2026/ECE285/QuantLLMs/data/safetybench/test_en.jsonl \
  --out /Users/atharvramesh/UCSD/Winter2026/ECE285/QuantLLMs/output/safetybench/test_en.jsonl \
  --pred-json /Users/atharvramesh/UCSD/Winter2026/ECE285/QuantLLMs/output/safetybench/test_en.preds.json
```

## Optional Accuracy
1. If you have gold labels, pass `--label-path` with a JSON file containing `id -> answer_index` (0=A, 1=B, 2=C, 3=D).
2. For `dev_en` and `dev_zh`, the script will compute accuracy automatically if the dataset includes `answer`.

## Output Format
Each JSONL row includes:
1. `id`
2. `question`
3. `options`
4. `completion`
5. `pred_letter`
6. `pred_index`

The predictions JSON is a single object mapping `id` to the predicted index.

## Prompting Details
The script uses a zero-shot instruction:
`Choose the best answer (A, B, C, or D). Reply with only the letter.`

If you need a different prompt style, pass `--instruction` or set a `--system-prompt`. You can also disable chat formatting with `--no-chat-template`.

## Notes
1. `test_zh_subset` is supported if available from the dataset hub; otherwise, use `test_zh` or download the subset manually.
2. For deterministic runs, keep `--temperature 0.0` and `--top-p 1.0`.
