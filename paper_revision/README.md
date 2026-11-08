# FORGE — results and code of the revised IEEE Access paper

This folder lets you (1) rebuild every result table of the paper from the
released results, and (2) rerun FORGE and the baselines.

```
results/                 inputs of Tables 5-7, 9-15 (small JSON / JSONL)
make_paper_tables.py     rebuilds the tables from results/ (no training, no LLM)
runs.tar.zst             per-run records (1.5 MB; unpack with: tar --zstd -xf runs.tar.zst)
code/
  forge_engine/          FORGE engine used for every run in the paper
  baselines/             the five AutoFE baselines, run as "algorithm + Raw"
  run_forge.sh           one FORGE run
  run_baseline.sh        one baseline run
  score.py               scores baselines and the Pure-LLM control on the test set
  fetch_openfe.sh        official OpenFE code (needed by the OpenFE baseline)
```

`runs.tar.zst` unpacks to `runs/` and holds the R0 proposals, all LLM requests
and responses, and the accepted feature specifications of every reported run.

## 1. Rebuild the tables

```bash
pip install numpy scipy
python3 make_paper_tables.py
```

## 2. Rerun

Requirements: Python 3.10, `pip install -r code/baselines/requirements.txt`
(LightGBM 4.6, XGBoost 3.2, CatBoost 1.2, scikit-learn 1.7, pandas 2.0,
DuckDB), the seven datasets as DuckDB tables (see the main FORGE repository,
`datasets/`), and the Codex CLI for LLM calls.

```bash
export FORGE_DATA_DIR=/path/to/duckdb  OTHERS_DATA_DIR=/path/to/duckdb

# FORGE, reusing the recorded R0 proposals of the paper (no R0 LLM call)
R0_PLAN=runs/main/credit_default/lgbm/seed11/r0_plan/model_factor_plan.json \
  code/run_forge.sh credit_default 11 lgbm          # test AP 0.5502016

# a baseline (algorithm + Raw)
FORGE_LLM_MODEL=gpt-5.6-sol code/run_baseline.sh credit_default caafe 1

# score baselines and the Pure-LLM control
python3 code/score.py --datasets credit_default \
  --baseline-runs runs/baselines --forge-runs runs/main --out scores
```

`runs/` is the unpacked per-run archive. Runs whose CoF stage makes no LLM
call (here Credit Default) reproduce exactly; runs with live CoF calls vary by
about 0.003 AP between reruns.

## Scripts

| script | usage | environment |
|---|---|---|
| `code/run_forge.sh` | `<dataset> <seed> <lgbm\|xgb\|cat> [out_dir]` | `FORGE_DATA_DIR`, `FORGE_LLM_MODEL`, optional `R0_PLAN` (reuse recorded R0 proposals); switches in `code/forge_engine/SWITCHES.md` |
| `code/run_baseline.sh` | `<dataset> <openfe\|caafe\|featllm\|llmfe\|octree> <repeat> [out_dir]` | `OTHERS_DATA_DIR`, `FORGE_LLM_MODEL` |
| `code/fetch_openfe.sh` | run once before the OpenFE baseline | -- |
| `code/score.py` | `--datasets ... --baseline-runs runs/baselines --forge-runs runs/main --out scores` | `FORGE_DATA_DIR` |
| `make_paper_tables.py` | `[--latex]` | -- |

## Settings

| item | value |
|---|---|
| split | 64/16/20 (I/V/T), chronological; Credit Default, Twitterbot stratified (seed 42) |
| learners | LightGBM, XGBoost, CatBoost, 400 trees, learning rate 0.05 (Table 3) |
| seeds | 11, 12, 13 |
| FORGE proposal model | gpt-5.5 (Codex CLI, reasoning effort medium) |
| baseline proposal model | gpt-5.6-sol |
| baselines | official code with our data loader and AP as validation metric, default settings; OCTree with 3 columns x 24 rounds |
| strongest baseline | best of the five baselines (Table 5) |
| Pure-LLM control | seed-11 R0 plan of FORGE, all proposals added, single fit |
