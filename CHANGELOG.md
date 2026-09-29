# Changelog

## 0.3.1

First public release.

- FORGE as described in the paper: routing rule, R0 free-form proposals with
  PSI/AdvAUC screening and shadow admission, Chain-of-Focus rounds with LLM
  templates, residual-score entry, trailing-window check and permutation exit,
  harm-only round validation, optional TE / residual-slice / recency experts,
  and the four-fold core fit on I+V used by S2 and S3.
- Learners: LightGBM, XGBoost and CatBoost with the paper's hyperparameters,
  or your own (`--learner module:function`).
- All parameters in YAML (`configs/`), annotated with the paper's sections,
  equations and tables.
- LLM proposals: recorded (replay), an OpenAI-compatible API, or any
  command-line tool such as Codex, Claude Code or Ollama.
- Loaders that rebuild the paper's seven dataset tables from their public
  sources.
- `python main.py` reproduces the paper's Credit Default / LightGBM / seed-11
  test AP (0.5502016) exactly.
- Licensed under the Apache License 2.0.
