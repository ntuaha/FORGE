#!/usr/bin/env bash
set -euo pipefail
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; E="$H/forge_engine"
DS="${1:?dataset}"; SEED="${2:?seed}"; LR="${3:?lgbm|xgb|cat}"
OUT="${4:-$H/../runs/forge_new/$DS/$LR/seed$SEED}"
case "$LR" in lgbm) SEAT=lgbm1 ;; xgb) SEAT=xgb1 ;; cat) SEAT=cat1 ;; *) exit 2 ;; esac
export L1_SEATS=$SEAT V34_PROBE_KIND=$LR CUDA_VISIBLE_DEVICES="" FORGE_V13_FIT_WORKERS="${FORGE_V13_FIT_WORKERS:-4}"
: "${FORGE_DATA_DIR:?set FORGE_DATA_DIR to the DuckDB folder}"
PY="${PY:-python3}"; COMMON=(--device cpu --l1 "$SEAT" --l2 2 --oof-folds 4 --seed "$SEED" --representation raw)
mkdir -p "$OUT"
(cd "$E/forge_lite" && $PY -m forge_v13.run_forge_v13 "$DS" --stage baseline --out "$OUT/baseline" "${COMMON[@]}")
if [ -n "${R0_PLAN:-}" ]; then mkdir -p "$OUT/r0_plan"; cp "$R0_PLAN" "$OUT/r0_plan/model_factor_plan.json"
else (cd "$E/forge_lite" && $PY -m forge_v13.free_llm_v34 "$DS" --out "$OUT/r0_plan" "${COMMON[@]}" --per-order 8 --cap 36); fi
(cd "$E/forge_lite" && $PY -m forge_v13.run_forge_v13 "$DS" --stage forge --baseline-dir "$OUT/baseline" \
  --out "$OUT/forge" "${COMMON[@]}" --max-rounds 3 --gate-block-weights 3,1,1 --lite-search-folds 4 --lite-v21 \
  --domain-context "$E/context/factor_families.md" --lite-v9 --bank-no-te --perceive-v2 --s4-funnel \
  --s4-entry noise_floor --s4-eval chrono --lite-prune-weight 0 --v34-router \
  --inject-factors "$OUT/r0_plan/model_factor_plan.json" --inject-mode gated_surface)
$PY -c "import json;print('test AP', json.load(open('$OUT/forge/summary.json'))['final_auprc'])"
