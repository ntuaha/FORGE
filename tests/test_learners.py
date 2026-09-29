"""Every learner runs the whole pipeline (R0, CoF, experts, rebuild) on the small demo dataset."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from forge import make_config, run_forge  # noqa: E402


class LearnerTests(unittest.TestCase):
    def test_all_learners_on_the_demo(self):
        for learner in ("lightgbm", "xgboost", "catboost", "examples.my_learner:make_model"):
            with self.subTest(learner=learner):
                result = run_forge(make_config("temporal_demo", learner=learner,
                                               **{"examples.my_learner:make_model": {"max_iter": 80}}))
                self.assertGreater(result.scores["cof"]["ap"], result.scores["baseline"]["ap"] - 0.05)
                self.assertTrue(any(r["round"] > 0 for r in result.rounds))      # CoF rounds ran


if __name__ == "__main__":
    unittest.main()
