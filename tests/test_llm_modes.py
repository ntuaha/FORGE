"""The command and api LLM modes end to end, without a real LLM."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forge import llm, make_config, run_forge  # noqa: E402

STUB = Path(__file__).resolve().parent / "stub_llm.py"


class LlmModeTests(unittest.TestCase):
    def test_command_mode_then_replay_from_cache(self):
        with tempfile.TemporaryDirectory() as cache:
            cfg = make_config("temporal_demo", llm_mode="command", llm_command=f"{sys.executable} {STUB}",
                              cache_dir=Path(cache))
            live = run_forge(cfg)
            self.assertTrue(live.scorecard)
            replay = run_forge({**cfg, "llm_mode": "replay"})       # answers now come from the cache
            self.assertEqual(live.scores, replay.scores)

    def test_api_mode_uses_the_endpoint(self):
        def endpoint(prompt, schema, cfg):          # an "endpoint" that answers like the stub tool
            answer = subprocess.run([sys.executable, str(STUB)], capture_output=True, text=True,
                                    input=prompt + "\nJSON schema:\n" + json.dumps(schema)).stdout
            return llm.parse_json(answer)

        with tempfile.TemporaryDirectory() as cache, mock.patch.dict("os.environ", {"FORGE_LLM_API_KEY": "k"}):
            cfg = make_config("temporal_demo", llm_mode="api", llm_api_url="http://stub", cache_dir=Path(cache))
            with mock.patch.object(llm, "call_api", side_effect=endpoint) as call:
                run_forge(cfg)
            self.assertGreaterEqual(call.call_count, 6)

    def test_settings_are_checked_before_training(self):
        with self.assertRaises(ValueError):
            llm.check_llm_settings(make_config("temporal_demo", llm_mode="command", llm_command=""))
        with mock.patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            llm.check_llm_settings(make_config("temporal_demo", llm_mode="api", llm_api_url="http://x"))

    def test_parse_json_accepts_fenced_and_embedded_answers(self):
        self.assertEqual(llm.parse_json('text ```json\n{"a": 1}\n``` more'), {"a": 1})
        self.assertEqual(llm.parse_json('Sure: {"a": {"b": 2}} done'), {"a": {"b": 2}})


if __name__ == "__main__":
    unittest.main()
