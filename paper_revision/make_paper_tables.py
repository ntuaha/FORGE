#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DS = ["fraudecom", "sparknov", "banksim", "creditcard", "vehicleloan", "credit_default", "twitterbot"]
LR = ["lgbm", "xgb", "cat"]
NAME = {"fraudecom": "Fraudecom", "sparknov": "Sparknov", "banksim": "Banksim", "creditcard": "Creditcard",
        "vehicleloan": "Vehicleloan", "credit_default": "Credit Default", "twitterbot": "Twitterbot"}
LEARNER = {"lgbm": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost"}


def load(name):
    return json.loads((RES / name).read_text())


def rows_text(title, header, rows):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(w[i]) for i, c in enumerate(r))
    return "\n".join([f"== {title}", line(header)] + [line(r) for r in rows]) + "\n"


def wins_tests(pairs, ref):
    d = [p["forge"] - p[ref] for p in pairs]
    by = defaultdict(list)
    for p, x in zip(pairs, d):
        by[p["dataset"]].append(x)
    dsm = [float(np.mean(v)) for v in by.values()]
    return (sum(x > 0 for x in d), float(np.median(d)), wilcoxon(d, method="exact").pvalue,
            sum(x > 0 for x in dsm), wilcoxon(dsm, method="exact").pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="also write LaTeX fragments to tables_tex/")
    args = ap.parse_args()
    out = []
    M = load("main_comparison.json")
    P = {(p["dataset"], p["learner"]): p for p in M["pairs"]}

    out.append(rows_text("Table 5  FORGE (mean ± SD over seeds 11-13) vs strongest baseline",
                         ["dataset", "learner", "FORGE", "best baseline", "method"],
                         [[NAME[d], LEARNER[l], f"{P[(d, l)]['forge']:.3f}±{P[(d, l)]['forge_psd']:.3f}",
                           f"{P[(d, l)]['best_ap']:.3f}", P[(d, l)]["best"]] for d in DS for l in LR]))
    out.append(rows_text("Table 6  FORGE vs Pure-LLM control",
                         ["dataset", "learner", "FORGE", "Pure-LLM", "delta"],
                         [[NAME[d], LEARNER[l], f"{P[(d, l)]['forge']:.3f}", f"{P[(d, l)]['control']:.3f}",
                           f"{P[(d, l)]['forge'] - P[(d, l)]['control']:+.3f}"] for d in DS for l in LR]))
    rows = []
    for name, ref in (("Pure-LLM control", "control"), ("Strongest baseline", "best_ap")):
        w, med, p, dw, dp = wins_tests(M["pairs"], ref)
        rows.append([name, f"{w}/21", f"{med:+.3f}", f"{p:.2e}", f"{dw}/7", f"{dp:.3f}"])
    out.append(rows_text("Table 10  paired exact Wilcoxon tests", ["reference", "wins", "median", "pair p",
                                                                  "dataset wins", "dataset p"], rows))

    R = load("proposer_ablation_sensitivity.json")
    pr = R["proposer"]
    names = pr["complete"]
    out.append(rows_text("Table 11  proposal LLM for R0 (LightGBM, mean ± SD over seeds 11-13)", ["dataset"] + names,
                         [[NAME[d]] + [f"{np.mean(pr['table'][n][d]):.3f}±{np.std(pr['table'][n][d]):.3f}" for n in names]
                          for d in DS]))
    out.append(f"   Friedman over 21 blocks: chi2={pr['friedman']['chi2']:.2f}, p={pr['friedman']['p']:.4f}\n")

    U = load("uncertainty_practical_temporal.json")
    Q = {(p["dataset"], p["learner"]): p for p in U["pairs"]}
    out.append(rows_text("Table 9  difference to strongest baseline with 95% bootstrap CI",
                         ["dataset", "learner", "delta", "95% CI"],
                         [[NAME[d], LEARNER[l], f"{Q[(d, l)]['delta']:+.3f}",
                           f"[{Q[(d, l)]['ci'][0]:+.3f}, {Q[(d, l)]['ci'][1]:+.3f}]"] for d in DS for l in LR]))
    dl = U["dataset_level"]
    out.append(f"   dataset level: {dl['wins']}/7 positive, mean {dl['mean']:+.3f}, cluster CI "
               f"[{dl['cluster_ci'][0]:+.3f}, {dl['cluster_ci'][1]:+.3f}], p={dl['p']:.3f}\n")
    keys = ["p_at_r50", "p_at_r80", "r_at_p50", "r_at_p80", "recall_top1"]
    out.append(rows_text("Table 13  operating points, FORGE/baseline (mean over learners)", ["dataset"] + keys,
                         [[NAME[d]] + [f"{np.mean([Q[(d, l)]['forge_ops'][k] for l in LR]):.3f}/"
                                       f"{np.mean([Q[(d, l)]['best_ops'][k] for l in LR]):.3f}" for k in keys]
                          for d in DS]))
    out.append(rows_text("Table 12  AP by chronological quarter of T, FORGE/baseline", ["dataset", "Q1", "Q2", "Q3", "Q4"],
                         [[NAME[d]] + [f"{a:.3f}/{b:.3f}" for a, b in zip(
                             np.mean([Q[(d, l)]["quarters"]["forge"] for l in LR], axis=0),
                             np.mean([Q[(d, l)]["quarters"]["best"] for l in LR], axis=0))]
                          for d in DS if "quarters" in Q[(d, "lgbm")]]))

    B = load("budget_cost.json")
    rows = [[m, B[m]["calls"] or 0, f"{np.floor(B[m]['tok_k'] * 10 + 0.5) / 10:.1f}" if B[m]["tok_k"] else "--", B[m]["proposals"], B[m]["selected"],
             f"{B[m]['wall_min']:.1f} ({B[m]['wall_max']:.1f})", B[m]["rss_gb"]]
            for m in ["openfe", "caafe", "featllm", "llmfe", "octree"]]
    fc = B["forge"]["cost"]["per_dataset"]
    rss = [r["peak_rss_gb"] for r in B["forge"]["peak_rss"]]
    rows.append(["FORGE", round(float(np.median([v["calls_mean"] for v in fc.values()]))),
                 round(float(np.median([v["tok_k_mean"] for v in fc.values()])), 1),
                 f"{B['forge']['r0_valid']:.0f}+{B['forge']['cof_candidates']:.0f}", B["forge"]["selected"],
                 f"{np.median([v['wall_min_mean'] for v in fc.values()]):.1f} "
                 f"({max(v['wall_min_max'] for v in fc.values()):.1f})", f"{min(rss):.1f}-{max(rss):.1f}"])
    out.append(rows_text("Table 14  budget and cost per run", ["method", "LLM calls", "tokens (k)", "candidates",
                                                                "selected", "wall min (max)", "memory GB"], rows))

    ref = R["ref"]
    C = R["configuration"]
    gain = [x - r for d in DS for x, r in zip(C[d]["Raw+CoF"], C[d]["Raw"])]
    rows = [["Raw (mean AP)"] + [f"{np.mean(C[d]['Raw']):.3f}" for d in DS] + [""],
            ["all components: gain over Raw"] + [f"{np.mean(C[d]['Raw+CoF']) - np.mean(C[d]['Raw']):+.3f}" for d in DS]
            + [f"{wilcoxon(gain, method='exact').pvalue:.2e}"]]
    for arm, r in R["ablation"].items():
        rows.append([f"without {arm}"] + ["=" if d not in r["executed"]
                                          else f"{np.mean(r['cells'][d]) - np.mean(ref[d]):+.3f}" for d in DS]
                    + [f"{r['p_runs']:.3f}"])
    rows.append(["S2 experts: gain over CoF core"] + ["=" if d == "creditcard" else
                 f"{np.mean(C[d]['Raw+CoF+Experts']) - np.mean(C[d]['Raw+CoF']):+.3f}" for d in DS] + [""])
    out.append(rows_text("Table 7  component analysis (LightGBM)", ["row"] + [NAME[d] for d in DS] + ["p"], rows))
    rows = [["reference AP"] + [f"{np.mean(ref[d]):.3f}" for d in DS] + [""]]
    for arm, r in R["sensitivity"].items():
        rows.append([arm] + ["=" if d not in r["executed"] else f"{np.mean(r['cells'][d]) - np.mean(ref[d]):+.3f}"
                             for d in DS] + [f"{r['p_runs']:.3f}"])
    TM = R["sensitivity_main"]
    for key, r in TM["rows"].items():
        rows.append([key + " (vs main runs)"] + ["=" if d not in r["executed"] else
                     f"{np.mean(r['cells'][d]) - np.mean(TM['ref'][d]):+.3f}" for d in DS] + [f"{r['p_runs']:.3f}"])
    out.append(rows_text("Table 15  threshold sensitivity (change in mean AP vs reference)",
                         ["variant"] + [NAME[d] for d in DS] + ["p"], rows))

    F = load("full_rerun.json")
    out.append("== Full rerun of all 21 pairs (Section VI-B)\n"
               f"   mean |rerun - main| = {F['rerun_minus_main']['mean_abs']:.3f}; wins vs baseline "
               f"{F['rerun_all']['best_ap']['wins']}/21, vs Pure-LLM {F['rerun_all']['control']['wins']}/21\n")
    text = "\n".join(out).replace("-0.000", "0.000").replace("+0.000", "0.000")
    print(text)
    if args.latex:
        d = HERE / "tables_tex"
        d.mkdir(exist_ok=True)
        (d / "all_tables.txt").write_text(text)
        print(f"written {d / 'all_tables.txt'}")


if __name__ == "__main__":
    main()
