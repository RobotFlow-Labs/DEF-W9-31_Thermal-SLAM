#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR=${1:-}
OUTPUT_DIR=${2:-outputs}

if [[ -z "${INPUT_DIR}" ]]; then
  echo "usage: $0 <input_dir> [output_dir]"
  exit 1
fi

uv run python -m anima_thermal_slam.infer \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}"
