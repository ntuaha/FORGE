"""Unit checks of data isolation and of the statistical decision rules."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from forge import make_config  # noqa: E402
from forge.data import split_dataset, validate_dataset, validation_blocks, validation_halves  # noqa: E402
from forge.datasets import make_temporal_demo  # noqa: E402
from forge.features import Surface, make_representation_plan, materialize_expression, matrix  # noqa: E402
from forge.history import entity_history, entity_history_library  # noqa: E402
from forge.scoring import admission_threshold, greedy_entry  # noqa: E402
from forge.screening import adversarial_auc, drift_table, psi, trailing_window_check  # noqa: E402
from forge.templates import depth_contract, expand_templates  # noqa: E402
from forge.validation import ValidationBudget, choose_configuration, harm_only_gate, paired_bootstrap  # noqa: E402


class PartitionTests(unittest.TestCase):
    def test_split_is_disjoint_and_test_opens_once(self):
        cfg = make_config("temporal_demo")
        frame = make_temporal_demo(cfg)
        part = split_dataset(frame, cfg)
        ids = [set(part.ids[k]) for k in ("I", "V", "T")]
        self.assertFalse(ids[0] & ids[1] or ids[1] & ids[2] or ids[0] & ids[2])
        self.assertEqual(len(set.union(*ids)), len(frame))
        self.assertNotIn("target", part.X_T)
        self.assertEqual([len(part.I), len(part.V), len(part.X_T)], [5120, 1280, 1600])
        part.vault.evaluate({"baseline": np.full(len(part.X_T), .5)})
        with self.assertRaises(RuntimeError):
            part.vault.evaluate({"baseline": np.full(len(part.X_T), .5)})

    def test_validation_blocks_are_3_1_1_and_contiguous(self):
        blocks = validation_blocks(4800)
        self.assertEqual([len(b) for b in blocks], [2880, 960, 960])
        np.testing.assert_array_equal(np.concatenate(blocks), np.arange(4800))
        halves = validation_halves(4800)
        self.assertEqual([len(h) for h in halves], [2400, 2400])

    def test_budget_consumes_blocks_in_order(self):
        budget = ValidationBudget(1000, make_config("temporal_demo"))
        block, regions = budget.next_regions(initial_round=True)
        self.assertEqual((block, [len(r) for r in regions]), (1, [300, 300]))
        block, regions = budget.next_regions(initial_round=False)
        self.assertEqual((block, [len(r) for r in regions]), (2, [200]))
        budget.next_regions(initial_round=False)
        self.assertTrue(budget.exhausted)
        with self.assertRaises(RuntimeError):
            budget.next_regions(initial_round=False)


class StatisticsTests(unittest.TestCase):
    def test_paired_bootstrap_uses_same_rows_and_sample_sd(self):
        y = np.tile([0, 1], 50)
        p = np.linspace(.1, .9, len(y))
        q = p + .01 * y
        rng = np.random.default_rng(17)
        differences = []
        for _ in range(40):
            rows = rng.integers(0, len(y), len(y))
            differences.append(average_precision_score(y[rows], q[rows]) - average_precision_score(y[rows], p[rows]))
        self.assertAlmostEqual(paired_bootstrap(y, p, q, 40, 17)["se"], np.std(differences, ddof=1))

    def test_harm_only_gate_rejects_clear_degradation_and_accepts_equal(self):
        y = np.tile([0, 1], 100)
        cfg = make_config("temporal_demo", bootstrap_repeats=40)
        regions = validation_halves(len(y))
        passed, _ = harm_only_gate(y, .1 + .8 * y, .9 - .8 * y, regions, cfg)
        self.assertFalse(passed)
        p = np.linspace(.1, .9, len(y))
        passed, _ = harm_only_gate(y, p, p.copy(), regions, cfg)
        self.assertTrue(passed)

    def test_psi_and_advauc(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=2000), rng.normal(size=2000)
        self.assertLess(psi(a, b), .05)
        self.assertAlmostEqual(adversarial_auc(a, b), .5, delta=.05)
        self.assertGreater(psi(a, b + 2), .25)
        self.assertGreater(adversarial_auc(a, b + 2), .9)

    def test_threshold_and_greedy_entry(self):
        self.assertAlmostEqual(admission_threshold(100), 1.6448536269514722 / 10)
        rng = np.random.default_rng(1)
        n = 3000
        x, signal, noise = rng.normal(size=(3, n))
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(x + 2 * signal)))).astype(int)
        oof = 1 / (1 + np.exp(-x))
        train = pd.DataFrame({"target": y, "x": x, "signal": signal, "noise": noise,
                              "signal_copy": signal * 2.0})
        cfg = make_config("temporal_demo")
        entered, audit = greedy_entry(train, ["noise", "signal", "signal_copy"], ["x"], y - oof, cfg)
        # The strongest direction enters first; its exact copy is then redundant.
        self.assertEqual([name for name, _ in entered][:1], ["signal"])
        self.assertNotIn("signal_copy", [name for name, _ in entered])
        # entered carries S_I, the first-pass score, which equals the first audit score.
        self.assertAlmostEqual(entered[0][1], audit[0]["score"])

    def test_trailing_window_rules_by_cycle(self):
        rng = np.random.default_rng(3)
        n = 4000
        signal = rng.normal(size=n)
        residual = np.where(np.arange(n) < .7 * n, 1., -1.) * signal + .1 * rng.normal(size=n)
        train = pd.DataFrame({"target": (rng.uniform(size=n) < .3).astype(int), "x": rng.normal(size=n),
                              "flip": signal})
        cfg = make_config("temporal_demo")
        # S_I > 0 but the association reverses in the last 30%: removed in every cycle.
        for cycle in (1, 2):
            kept, _ = trailing_window_check(train, [("flip", 0.3)], ["x"], residual, cycle, cfg)
            self.assertEqual(kept, [])
        # The same sign is kept in cycle 1 and, being strong, in cycle 2 as well.
        kept, _ = trailing_window_check(train, [("flip", -0.3)], ["x"], residual, 2, cfg)
        self.assertEqual(kept, ["flip"])

    def test_core_selection_is_highest_v_ap(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 0])
        states = [{"features": [], "V_prediction": np.linspace(0, 1, 8)},
                  {"features": ["a"], "V_prediction": y + 0.01 * np.arange(8)},
                  {"features": ["a", "b"], "V_prediction": 1 - y}]
        state, table = choose_configuration(states, y, make_config("temporal_demo"))
        self.assertEqual(state["features"], ["a"])
        self.assertEqual(int(table["selected"].sum()), 1)

    def test_routing_shift_uses_unseen_categories_for_categoricals(self):
        cfg = make_config("x", label="target")
        I = pd.DataFrame({"target": [0, 1] * 100, "cat": ["a", "b"] * 100, "num": np.arange(200.)})
        V = pd.DataFrame({"target": [0, 1] * 100, "cat": ["a"] * 150 + ["new"] * 50, "num": np.arange(200.)})
        table = drift_table(I, V, cfg).set_index("feature")
        self.assertTrue(table.loc["cat", "drifted"])          # 25% unseen categories
        self.assertAlmostEqual(table.loc["cat", "novelty"], .25)


class SafetyTests(unittest.TestCase):
    def test_expressions_cannot_read_labels_or_do_io(self):
        frame = pd.DataFrame({"x": [1., 2.], "target": [0, 1]})
        for expression in ("df['target']", "np.load('x')", "df['x'].mean()", "df['x'].__class__",
                           "__import__('os')", "df"):
            with self.subTest(expression=expression), self.assertRaises((ValueError, SyntaxError)):
                materialize_expression(frame, expression, "target")
        np.testing.assert_allclose(materialize_expression(frame, "np.log1p(df['x'])", "target"),
                                   np.log1p([1., 2.]).astype(np.float32))

    def test_history_excludes_same_time_and_future(self):
        cfg = make_config("temporal_demo")
        frame = pd.DataFrame({"event_time": [1, 2, 2, 3], "customer": [1] * 4, "income": [10., 20., 900., 40.]})
        spec = {"entity": "customer", "value": "income", "agg": "mean", "window": 0, "shift": 1}
        before = entity_history(frame, None, spec, cfg)
        np.testing.assert_allclose(before, [0, 10, 10, 310])
        frame.loc[3, "income"] = 1e9
        np.testing.assert_array_equal(before[:3], entity_history(frame, None, spec, cfg)[:3])
        count = entity_history(frame, None, {**spec, "agg": "count"}, cfg)
        np.testing.assert_allclose(count, [0, 1, 1, 3])
        np.testing.assert_allclose(entity_history(frame, None, {**spec, "agg": "time_since"}, cfg), [-1, 1, 1, 1])
        np.testing.assert_allclose(entity_history(frame, None, {**spec, "agg": "last", "window": 1, "shift": 2},
                                                  cfg), [0, 0, 0, 20])
        with self.assertRaises(ValueError):
            entity_history(frame, None, {**spec, "shift": 0}, cfg)
        with self.assertRaises(ValueError):
            entity_history(frame.assign(target=0), None, {**spec, "value": "target"}, cfg)

    def test_date_strings_are_parsed_as_time(self):
        cfg = make_config("x", label="target", time_col="t", entity_cols=["c"])
        frame = pd.DataFrame({"t": ["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"] * 150,
                              "c": 1, "target": [0, 1] * 300, "x": 1.0})
        frame = validate_dataset(frame, cfg).iloc[:4]
        spec = {"entity": "c", "agg": "count", "value": ""}
        np.testing.assert_allclose(entity_history(frame, None, spec, cfg), [0, 1, 1, 3])
        np.testing.assert_allclose(entity_history(frame, None, {**spec, "agg": "time_since"}, cfg),
                                   [-1, 86400, 86400, 86400])

    def test_yaml_overrides_merge_nested_settings(self):
        cfg = make_config("credit_default", lightgbm={"n_estimators": 80})
        self.assertEqual(cfg["lightgbm"]["n_estimators"], 80)
        self.assertEqual(cfg["lightgbm"]["num_leaves"], 63)          # untouched keys keep their defaults

    def test_library_is_empty_without_entity_columns(self):
        cfg = make_config("x", label="target")
        frame = pd.DataFrame({"target": [0, 1] * 50, "a": np.arange(100.), "b": np.arange(100) % 7})
        self.assertEqual(entity_history_library(frame, cfg, make_representation_plan(frame, cfg)), {})
        cfg = make_config("temporal_demo")
        demo = make_temporal_demo(cfg)
        library = entity_history_library(demo, cfg, make_representation_plan(demo, cfg))
        self.assertEqual({spec["agg"] for spec in library.values()}, {"count", "repeat", "time_since", "deviation"})

    def test_template_expansion_respects_the_depth_contract(self):
        cfg = make_config("temporal_demo")
        raw = ["income", "debt", "x1", "x2"]
        pointwise = {"name": "ratio", "kind": "pointwise", "mechanism": "", "form": "{A}/(np.abs({B})+1.0)",
                     "slots": [{"slot": "A", "columns": ["*"]}, {"slot": "B", "columns": ["*", "target"]}],
                     "entities": [], "values": [], "aggregations": [], "windows": [], "shift": 1}
        aggregate = {"name": "hist", "kind": "aggregate", "mechanism": "", "form": "", "slots": [],
                     "entities": ["customer"], "values": ["debt"], "aggregations": ["mean", "count"],
                     "windows": [0, 3], "shift": 1}
        specs, audit = expand_templates([pointwise, aggregate], depth_contract(1, raw, {}), raw, ["customer"], cfg)
        self.assertEqual(sum(s["executor"] == "pointwise" for s in specs.values()), 12)   # 4 x 3 ordered pairs
        self.assertEqual(sum(s["executor"] == "entity_history" for s in specs.values()), 4)
        self.assertIn("target", audit[0]["dropped_columns"])
        # Cycle 2: only cycle-1 features; aggregates are not allowed.
        specs, audit = expand_templates([pointwise, aggregate], depth_contract(2, raw, {1: ["f1", "f2"]}), raw,
                                        ["customer"], cfg)
        self.assertEqual(len(specs), 2)
        self.assertEqual(audit[1]["reason"], "aggregate templates are only allowed in cycle 1")
        # Cycle 3: at least one slot binds to a cycle-2 feature.
        specs, _ = expand_templates([pointwise], depth_contract(3, raw, {1: ["f1"], 2: ["g1"]}), raw, [], cfg)
        self.assertTrue(all("g1" in s["expression"] for s in specs.values()))
        self.assertEqual(len(specs), 10)                    # g1 with 5 first-order columns, both orders
        aggregate["shift"] = 0
        specs, audit = expand_templates([aggregate], depth_contract(1, raw, {}), raw, ["customer"], cfg)
        self.assertEqual(specs, {})

    def test_validation_labels_do_not_change_the_search_representation(self):
        cfg = make_config("temporal_demo")
        part = split_dataset(make_temporal_demo(cfg), cfg)
        Vx = part.V.drop(columns="target")
        plan = make_representation_plan(part.I, cfg)
        first = Surface(part.I, Vx, plan, {}, cfg)
        flipped = part.V.copy()
        flipped["target"] = 1 - flipped["target"]
        second = Surface(part.I, flipped.drop(columns="target"), plan, {}, cfg)
        cols = [c for c in first.apply]
        np.testing.assert_array_equal(matrix(first.apply, cols), matrix(second.apply, cols))


if __name__ == "__main__":
    unittest.main()
