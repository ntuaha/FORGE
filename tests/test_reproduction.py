"""End-to-end reproduction of the published Credit Default / LightGBM / seed-11 result.

Slow (about 20 s with 8 threads plus the download). Skipped when
``FORGE_SKIP_SLOW=1`` or when the UCI data cannot be obtained.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forge import make_config, run_forge  # noqa: E402
from forge.datasets import download_credit_default  # noqa: E402

EXPECTED = {"baseline": 0.5482592133059769, "cof": 0.5502016247535624, "cof_experts": 0.5502016247535624}
PAPER_AP = 0.5502016


@unittest.skipIf(os.environ.get("FORGE_SKIP_SLOW") == "1", "slow test disabled by FORGE_SKIP_SLOW=1")
class CreditDefaultReproduction(unittest.TestCase):
    def test_seed11_reproduces_published_ap(self):
        cfg = make_config("credit_default", seed=11)
        try:
            download_credit_default(cfg["data_dir"])
        except OSError as exc:  # no network and no cached copy
            self.skipTest(f"UCI Credit Default is not available: {exc}")
        result = run_forge(cfg)
        for key, value in EXPECTED.items():
            self.assertLess(abs(result.scores[key]["ap"] - value), 1e-9, key)
        self.assertLess(abs(result.scores["cof"]["ap"] - PAPER_AP), 1e-6)
        self.assertTrue(result.router["run_r0"])
        self.assertFalse(result.router["run_cof"])
        self.assertEqual(len(result.rounds), 1)
        self.assertEqual(result.accepted_features, ["free2_pay0_te_age_te_product", "free3_pay0_age_limit_norm",
                                                    "free4_pay_delay_positive_hhi", "free5_pay_delay_tail_burden"])


if __name__ == "__main__":
    unittest.main()
