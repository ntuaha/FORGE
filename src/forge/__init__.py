"""FORGE: Feature Optimization via Recursive Generation and Evaluation.

Reference implementation accompanying the paper by Cheng-Yu Lin and
Jyh-Shing Roger Jang (National Taiwan University).

Modules, in the order of the method (Algorithm 1 in ``pipeline``):

============  ==============================================================
config        YAML configuration (configs/default.yaml, configs/datasets/)
datasets      loaders of the paper datasets and of your CSV files
data          I/V/T partition, validation blocks/halves, test vault
features      learner representation, safe pointwise expressions
history       causal entity-history features (library, aggregate templates)
learners      LightGBM, XGBoost, CatBoost and four-fold OOF predictions
llm           R0 and CoF prompts; replay, API and command-line LLM modes; cache
templates     CoF template expansion under the depth contract, scorecard
screening     S1.1.2 PSI / AdvAUC screen; S1.1.3 (ii) trailing-window check; shift table
scoring       S1.1.3 (CoF): residual score S, threshold tau, greedy entry
admission     S1.1.3 (R0): shadow admission; CoF permutation exit
validation    S1.2 harm-only gate, S2.2 benefit check, CoF core selection
experts       S2: target-encoding, residual-slice and recency experts
pipeline      routing rule, R0, CoF rounds/cycles, S2, S3 and ``run_forge``
============  ==============================================================

>>> from forge import make_config, run_forge
>>> result = run_forge(make_config("credit_default", seed=11))
>>> result.scores["cof"]["ap"]
"""

from .config import make_config
from .pipeline import ForgeResult, run_forge

__all__ = ["make_config", "run_forge", "ForgeResult"]
__version__ = "0.3.1"
