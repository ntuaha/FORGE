# Environment switches

Every switch defaults to the configuration of the main runs when unset.

| variable | effect | used in |
|---|---|---|
| FORGE_LLM_MODEL | proposal model for R0 and the in-round CoF calls (default: `config.yaml`, gpt-5.5) | Tables 7, 11, 15; Sections VI-B, VII |
| FORGE_LLM_BACKEND, FORGE_LLM_API_URL, FORGE_LLM_API_KEY | `codex` (default) or an OpenAI-compatible endpoint | Table 11 (gemma-4-26B) |
| FORGE_NO_RESIDUAL=1 | centered target replaces the OOF residual in scoring and prompt evidence | Table 7 |
| FORGE_ABL_NO_RECURSION=1 | one proposal cycle per CoF round | Table 7 |
| FORGE_ABL_NO_TW=1, FORGE_TW_FRAC | disable or resize the trailing-window check (default 0.30) | Tables 7, 15 |
| FORGE_ABL_NO_SHADOW=1 | R0 admits every screened factor | Table 7 |
| FORGE_ABL_NO_SCREEN=1, FORGE_PSI_MAX, FORGE_ADV_BAND | disable or change the R0 PSI and AdvAUC screen | Tables 7, 15 |
| FORGE_HARM_K | round gate k in Delta >= -k SE (1e9 never rejects) | Tables 7, 15 |
| FORGE_ENTRY_ALPHA | alpha in tau = z_{1-alpha} / sqrt(n+) | Table 15 |
| FORGE_KEEP_FRAC | fraction of the training partition kept (default 1) | -- |
