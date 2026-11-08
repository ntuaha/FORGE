#!/usr/bin/env bash
set -euo pipefail
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS="${1:?dataset}"; M="${2:?method}"; R="${3:-1}"; OUT="${4:-$H/../runs/baselines_new/$DS/$M/r$R}"
declare -A S=([openfe]=openfe_official.py [caafe]=caafe_adapted.py [featllm]=featllm_adapted.py \
              [llmfe]=llmfe.py [octree]=octree_adapted.py)
export PYTHONPATH="$H/baselines" BASELINE_REPRESENTATION=e0 CUDA_VISIBLE_DEVICES=""
mkdir -p "$OUT"
(cd "$H/baselines" && ${PY:-python3} "methods/${S[$M]}" "$DS" --out "$OUT" --repeat "$R" --no-finalize)
