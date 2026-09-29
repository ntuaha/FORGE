"""Run FORGE from the command line.

    python main.py                                   # the paper's Credit Default / LightGBM / seed-11 run
    python main.py --learner xgboost --seed 13       # another learner or seed
    python main.py --dataset sparknov                # another paper dataset (files in data/, see README)
    python main.py --config my.yaml                  # any setting from configs/default.yaml

Every parameter comes from configs/default.yaml and configs/datasets/<name>.yaml;
options given here override them. For the paper's reference run (Credit
Default, LightGBM, seed 11, published settings, replayed proposals) the test
AP is compared with the published value, and the exit code is non-zero if
they differ by more than 1e-6. Other results of the paper are in its
Tables 5-9.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from forge import make_config, run_forge  # noqa: E402

TOLERANCE = 1e-6
REFERENCE_SEED = 11
REFERENCE_AP = 0.5502016        # published FORGE test AP, Credit Default / LightGBM / seed 11
RUN_ONLY = {"seed", "threads", "data_dir", "cache_dir", "data_files", "llm_mode", "llm_command", "llm_api_url",
            "llm_model", "llm_api_key_env", "llm_timeout_seconds"}
def same_as_paper(cfg: dict) -> bool:
    """True if every method setting equals the published Credit Default / LightGBM configuration."""
    reference = make_config("credit_default")
    return all(cfg.get(k) == v for k, v in reference.items() if k not in RUN_ONLY)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="YAML file with any settings (see configs/)")
    parser.add_argument("--dataset", help="a name in configs/datasets/ (default: credit_default)")
    parser.add_argument("--learner", help="lightgbm | xgboost | catboost | module:function (your own learner)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--llm-mode", choices=["replay", "api", "command"])
    parser.add_argument("--llm-command", help='e.g. "codex exec --skip-git-repo-check --output-schema {schema} '
                                              '-o {output} -" (implies --llm-mode command)')
    parser.add_argument("--data-file", type=Path, action="append", help="data file(s), overriding data_files")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--output", type=Path, help="directory for report/metrics/features/predictions")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")

    overrides = {k: v for k, v in {"learner": args.learner, "seed": args.seed, "llm_mode": args.llm_mode,
                                   "threads": args.threads}.items() if v is not None}
    if args.llm_command:
        overrides.update(llm_mode=args.llm_mode or "command", llm_command=args.llm_command)
    if args.data_file:
        overrides["data_files"] = [str(p.resolve()) for p in args.data_file]
    try:
        cfg = make_config(args.dataset, args.config, **overrides)
        result = run_forge(cfg)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.output:
        result.save(args.output)

    s = result.scores
    r0 = next((r for r in result.rounds if r["round"] == 0), None)
    source = r0["proposal_audit"][0].get("source") if r0 else None
    exact = source == "frozen_llm_proposals" and cfg["seed"] == REFERENCE_SEED and same_as_paper(cfg)
    paper = REFERENCE_AP if exact else None
    print()
    print(f"{cfg['dataset']}, {cfg['learner']}, seed {cfg['seed']}, LLM: {cfg['llm_mode']}")
    print(f"accepted features: {result.accepted_features}")
    print(f"{'configuration':<26}{'test AP':>20}{'paper':>12}{'|diff|':>11}")
    for name, key, ref in [("Raw baseline (this code)", "baseline", None), ("FORGE (Raw+CoF)", "cof", paper),
                           ("FORGE (Raw+CoF+Experts)", "cof_experts", None)]:
        ref_txt = f"{ref:.7f}" if ref is not None else "-"
        diff_txt = f"{abs(s[key]['ap'] - ref):.1e}" if ref is not None else "-"
        print(f"{name:<26}{s[key]['ap']:>20.16f}{ref_txt:>12}{diff_txt:>11}")
    if exact:
        if abs(s["cof"]["ap"] - paper) > TOLERANCE:
            print(f"MISMATCH: the reproduced FORGE AP differs from the paper by more than {TOLERANCE}")
            return 1
        print("OK: the published FORGE test AP is reproduced.")
    else:
        print("Compare with Tables 5-9 of the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
