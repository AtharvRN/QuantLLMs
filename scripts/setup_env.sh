#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v uv >/dev/null 2>&1; then
  uv venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -r requirements.txt
else
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found. Please install Python 3.10+ and retry." >&2
    exit 1
  fi
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

echo "\nEnvironment ready."
echo "Next: export HF_TOKEN=... (Hugging Face token for gated models)"
echo "Then: python scripts/prepare_prompts.py"
echo "And:  python scripts/run_eval.py --smoke"
