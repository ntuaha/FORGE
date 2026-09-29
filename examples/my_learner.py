"""Example of your own learner: scikit-learn's HistGradientBoostingClassifier.

    python main.py --learner examples.my_learner:make_model

Settings can be given in a YAML file under the same key as the learner:

    learner: "examples.my_learner:make_model"
    "examples.my_learner:make_model": {max_iter: 300, learning_rate: 0.05}
"""

from sklearn.ensemble import HistGradientBoostingClassifier


def make_model(params: dict, cfg: dict):
    """Return an unfitted classifier with fit(X, y) and predict_proba(X); X may contain NaN."""
    return HistGradientBoostingClassifier(**{"max_iter": 400, "learning_rate": 0.05, **params},
                                          random_state=cfg["seed"])
