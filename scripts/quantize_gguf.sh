#!/usr/bin/env bash
set -euo pipefail

# Quantize a HF model to GGUF using llama.cpp tooling.
# Requires:
#   - external/llama.cpp cloned and built (llama-quantize binary)
#   - HuggingFace model available locally
#
# Example:
#   scripts/quantize_gguf.sh meta-llama/Meta-Llama-3.1-8B-Instruct \
#     models/llama3.1-8b-instruct \
#     output/gguf \
#     Q4_K_M

MODEL_ID="${1:-}"
MODEL_DIR="${2:-}"
OUT_DIR="${3:-}"
QUANT="${4:-Q4_K_M}"

if [[ -z "${MODEL_ID}" || -z "${MODEL_DIR}" || -z "${OUT_DIR}" ]]; then
  echo "Usage: scripts/quantize_gguf.sh <model_id> <model_dir> <out_dir> [Q4_K_M]"
  exit 1
fi

LLAMA_DIR="external/llama.cpp"
CONVERT_PY="${LLAMA_DIR}/convert_hf_to_gguf.py"
QUANT_BIN="${LLAMA_DIR}/llama-quantize"

if [[ ! -f "${CONVERT_PY}" ]]; then
  echo "Missing ${CONVERT_PY}. Clone llama.cpp into external/llama.cpp."
  exit 1
fi
if [[ ! -x "${QUANT_BIN}" ]]; then
  echo "Missing ${QUANT_BIN}. Build llama.cpp (make GGML_CUDA=1)."
  exit 1
fi

mkdir -p "${MODEL_DIR}" "${OUT_DIR}"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Downloading model to ${MODEL_DIR}..."
  huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_DIR}" --local-dir-use-symlinks False
fi

FP16_GGUF="${OUT_DIR}/model-f16.gguf"
Q_GGUF="${OUT_DIR}/model-${QUANT}.gguf"

echo "Converting HF -> GGUF (f16)..."
python "${CONVERT_PY}" "${MODEL_DIR}" --outfile "${FP16_GGUF}" --outtype f16

echo "Quantizing to ${QUANT}..."
"${QUANT_BIN}" "${FP16_GGUF}" "${Q_GGUF}" "${QUANT}"

echo "Wrote ${Q_GGUF}"
