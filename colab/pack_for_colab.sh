#!/usr/bin/env bash
# Zip the project for Colab upload (excludes datasets/results/.git caches).
# Run on your Mac, then upload the zip in the notebook.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$HOME/Desktop/kmip-adaptive-k-colab.zip}"
cd "$ROOT"
rm -f "$OUT"
zip -r "$OUT" . \
  -x "./.git/*" \
  -x "./datasets/*" \
  -x "./results/*" \
  -x "./micromamba/*" \
  -x "./__pycache__/*" \
  -x "*/__pycache__/*" \
  -x "*.pyc" \
  -x "./.cursor/*" \
  -x "./profiling-experiment/.cache/*" \
  -x "./colab/*.zip" \
  -x "*.zip"
echo "Wrote $OUT ($(du -h "$OUT" | awk '{print $1}'))"
echo "Upload this zip in the Colab notebook (Cell: Get the code)."
