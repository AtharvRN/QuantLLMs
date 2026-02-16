# Llama 3.1 Quantization Comparison (AWQ, GPTQ, GGUF)

This workflow compares FP16 vs quantized variants of `meta-llama/Meta-Llama-3.1-8B-Instruct` on SafetyBench.

## Notes on vLLM Support
vLLM supports AWQ and GPTQ quantized models and also supports GGUF (single-file) models, which are commonly produced by llama.cpp tooling. GGUF support is marked experimental in vLLM docs and requires a single `.gguf` file. citeturn0search0turn0search1

## Prereqs
1. Access to the Llama 3.1 checkpoint (HuggingFace token and acceptance of model terms).
2. vLLM installed.
3. Quantization deps:
```
pip install -r requirements-quant.txt
```
4. GGUF tools: clone and build llama.cpp:
```
git clone https://github.com/ggml-org/llama.cpp external/llama.cpp
cd external/llama.cpp
make GGML_CUDA=1
cd -
```

## Step 1: Download SafetyBench data and labels
```
python scripts/download_safetybench.py --split test_en --out data/safetybench/test_en.jsonl
git clone --depth 1 https://github.com/thu-coai/SafetyBench.git external/SafetyBench
mkdir -p data/safetybench/opensource_data
cp -R external/SafetyBench/opensource_data/* data/safetybench/opensource_data/
```

## Step 2: Quantize AWQ
```
python scripts/quantize_awq.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --out models/llama3.1-8b-instruct-awq
```

## Step 3: Quantize GPTQ
```
python scripts/quantize_gptq.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --out models/llama3.1-8b-instruct-gptq
```

## Step 4: Quantize GGUF (Q4_K_M)
```
bash scripts/quantize_gguf.sh \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  models/llama3.1-8b-instruct \
  output/gguf \
  Q4_K_M
```

## Step 5: Run SafetyBench (FP16 + quantized)
FP16:
```
python scripts/run_safetybench_vllm.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --split test_en \
  --input-jsonl data/safetybench/test_en.jsonl \
  --out output/safetybench/fp16.jsonl \
  --pred-json output/safetybench/fp16.preds.json
```

AWQ:
```
python scripts/run_safetybench_vllm.py \
  --model models/llama3.1-8b-instruct-awq \
  --quantization awq \
  --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct \
  --split test_en \
  --input-jsonl data/safetybench/test_en.jsonl \
  --out output/safetybench/awq.jsonl \
  --pred-json output/safetybench/awq.preds.json
```

GPTQ:
```
python scripts/run_safetybench_vllm.py \
  --model models/llama3.1-8b-instruct-gptq \
  --quantization gptq \
  --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct \
  --split test_en \
  --input-jsonl data/safetybench/test_en.jsonl \
  --out output/safetybench/gptq.jsonl \
  --pred-json output/safetybench/gptq.preds.json
```

GGUF:
```
python scripts/run_safetybench_vllm.py \
  --model output/gguf/model-Q4_K_M.gguf \
  --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct \
  --split test_en \
  --input-jsonl data/safetybench/test_en.jsonl \
  --out output/safetybench/gguf_q4km.jsonl \
  --pred-json output/safetybench/gguf_q4km.preds.json
```

## Step 6: Score and compare
```
python scripts/score_safetybench.py \
  --labels data/safetybench/opensource_data/test_answers_en.json \
  --pred-json output/safetybench/fp16.preds.json \
  --name fp16 \
  --pred-json-b output/safetybench/awq.preds.json \
  --name-b awq \
  --out output/safetybench/compare_fp16_vs_awq.json
```

Repeat for GPTQ and GGUF.
