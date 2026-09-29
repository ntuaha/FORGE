# Assets

`credit_default_r0_proposals.json` holds the 34 R0 free-form feature proposals
returned by `gpt-5.5` for UCI Credit Default in the run that produced the
published LightGBM / seed-11 result (five calls: unary, binary, ternary,
related-column and complement features). Only the name, expression, family,
description and executor of each proposal are stored; `proposals_sha256` is
checked on load.

`main.py` replays these proposals instead of calling an LLM. Everything else
(materialisation, screening, admission, validation, experts, the final
rebuild and the test evaluation) is recomputed. The file is used only when
the training partition matches the published configuration (Credit Default,
split seed 42, LightGBM with 400 trees / 63 leaves, 4 OOF folds, seed 11);
the order of the proposals matters because it fixes the random stream of the
shadow features.
