from __future__ import annotations

import os
from pathlib import Path


def _find_db_dir() -> Path:
    configured = os.environ.get("FORGE_DATA_DIR")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(
            f"FORGE_DATA_DIR does not exist or is not a directory: {candidate}")
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "datasets" / "duckdb_v2"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "cannot locate datasets/duckdb_v2; set FORGE_DATA_DIR explicitly")


DB_DIR = _find_db_dir()


DATASETS: dict[str, dict] = {
    "banksim": dict(
        name="banksim", table="raw", label="fraud", time_col="step",
        drop=["zipcodeori", "zipmerchant"],
        desc=("BankSim synthetic bank payments (595k rows, 1.2% fraud, "
              "customer/merchant/category entities). Chronological "
              "evaluation by simulation step."),
    ),
    "baf": dict(
        name="baf", table="base", label="fraud_bool", time_col="month",
        drop=[],
        desc=("Bank Account Fraud suite Base variant (NeurIPS 2022, 1M "
              "applications, 1.1% fraud, 8 months). Chronological "
              "evaluation by application month."),
    ),
    "ibmcard": dict(
        name="ibmcard", table="raw_sub", label="is_fraud", time_col="ts",
        drop=[],
        desc=("IBM TabFormer synthetic card transactions, chronological "
              "tail subsample of 2M rows (2019-2020, 0.1% fraud, "
              "card/merchant entities). Evaluation by timestamp."),
    ),
    "vehicleloan": dict(
        name="vehicleloan", table="raw", label="loan_default",
        time_col="disbursal_date",
        drop=["uniqueid", "disbursal_date", "date_of_birth",
              "average_acct_age", "credit_history_length",
              "employment_type", "perform_cns_score_description"],
        desc=("Vehicle loan default (India L&T, 233k loans, about 21.6% "
              "default). Chronological evaluation by disbursal date."),
    ),
    "credit_default": dict(
        name="credit_default", table="raw",
        label="default.payment.next.month", time_col=None, drop=["id"],
        desc=("UCI Taiwan credit-card default: one row per client; stratified "
              "random outer and search-validation splits."),
    ),
    "ieeecis": dict(
        name="ieeecis", table="raw", label="isfraud",
        time_col="transactiondt", drop=["transactionid"],
        desc=("IEEE-CIS transaction fraud. Chronological outer test and "
              "forward search-validation splits; high-missingness columns kept."),
        final_no_xgb=True,
    ),
    "creditcard": dict(
        name="creditcard", table="raw", label="class",
        time_col="time", drop=[],
        desc=("ULB credit-card fraud with anonymized PCA columns. "
              "Chronological evaluation by Time."),
    ),
    "fraudecom": dict(
        name="fraudecom", table="raw", label="class",
        time_col="purchase_ts", drop=["user_id"],
        desc=("E-commerce fraud. Chronological evaluation by purchase time."),
    ),
    "sparknov": dict(
        name="sparknov", table="raw", label="is_fraud",
        time_col="unix_time", drop=["column00", "trans_num",
                                      "trans_date_trans_time"],
        desc=("Sparkov simulated transaction fraud. Chronological evaluation; "
              "CSV row index and unique transaction ID excluded. UTC hour, "
              "weekday and month bases are available for harmonic proposals."),
        final_no_xgb=True,
    ),
    "sparknov_trainonly": dict(
        name="sparknov_trainonly", table="raw", label="is_fraud",
        time_col="unix_time", drop=["column00", "trans_num",
                                      "trans_date_trans_time"],
        desc=("Sparkov fraudTrain-only benchmark: 1,296,675 simulated "
              "transactions, chronological 80/20 outer split matching the "
              "historical FORGE population. CSV row index and unique "
              "transaction ID excluded; UTC hour, weekday and month bases "
              "are available for harmonic proposals."),
        final_no_xgb=True,
    ),
    "sparknov_external": dict(
        name="sparknov_external", table="train_raw",
        external_test_table="test_raw", label="is_fraud",
        time_col="unix_time", drop=["column00", "trans_num",
                                      "trans_date_trans_time"],
        desc=("Sparkov strict external benchmark: fraudTrain is the complete "
              "training/search population and fraudTest is an immutable "
              "chronologically later final test. IDs excluded; UTC hour, "
              "weekday and month bases support harmonic proposals."),
        final_no_xgb=True,
    ),
    "twitterbot": dict(
        name="twitterbot", table="raw", label="is_bot",
        time_col=None, drop=["id", "location"],
        desc=("Twitter account bot detection: one row per account; stratified "
              "random outer and search-validation splits."),
    ),
    "paysim": dict(
        name="paysim", table="raw", label="isfraud",
        time_col="step", drop=["nameorig", "namedest", "isflaggedfraud"],
        desc=("PaySim mobile-money simulation. Chronological evaluation by step."),
        final_no_xgb=True,
        final_grid=[
            dict(num_leaves=63, learning_rate=0.1, n_estimators=200),
            dict(num_leaves=127, learning_rate=0.05, n_estimators=400),
        ],
    ),
    "talkingdata": dict(
        name="talkingdata", table="raw", label="label",
        time_col="click_time", drop=[],
        desc=("Full TalkingData click-attribution training table. "
              "Chronological evaluation; search and finalization run alone."),
        lgbm_override=dict(
            n_estimators=200, learning_rate=0.1, num_leaves=63, n_jobs=16,
        ),
        final_grid=[
            dict(num_leaves=63, learning_rate=0.1, n_estimators=200),
            dict(num_leaves=127, learning_rate=0.08, n_estimators=300),
            dict(num_leaves=255, learning_rate=0.05, n_estimators=400),
        ],
        final_no_xgb=True,
    ),

    "bank_marketing": dict(
        name="bank_marketing", table="raw", label="y", time_col=None,
        drop=["duration"],
        desc=("UCI Bank Marketing: 45,211 phone contacts of a Portuguese "
              "bank, 11.7% subscribe a term deposit; post-outcome call "
              "duration excluded; stratified random splits."),
    ),
    "online_shoppers": dict(
        name="online_shoppers", table="raw", label="revenue", time_col=None,
        drop=[],
        desc=("UCI Online Shoppers Purchasing Intention: 12,330 sessions, "
              "15.5% end in a purchase; stratified random splits."),
    ),
    "magic_gamma": dict(
        name="magic_gamma", table="raw", label="is_gamma", time_col=None,
        drop=[],
        desc=("UCI MAGIC Gamma Telescope: 19,020 simulated air showers, "
              "64.8% gamma signal; stratified random splits."),
    ),
    "drybean": dict(
        name="drybean", table="raw", label="is_barbunya", time_col=None,
        drop=[],
        desc=("UCI Dry Bean: 13,611 segmented bean images, 16 shape "
              "descriptors, one-vs-rest on the BARBUNYA variety (9.7%); "
              "stratified random splits."),
    ),
    "occupancy": dict(
        name="occupancy", table="raw", label="occupancy",
        time_col="ts", drop=["ts", "date"],
        desc=("UCI Occupancy Detection: 20,560 minutes of room temperature, "
              "humidity, light and CO2, 23.1% occupied; the three shipped "
              "files are concatenated in true timestamp order (their file "
              "names do not order them) and evaluated chronologically."),
    ),
    "htru2": dict(
        name="htru2", table="raw", label="is_pulsar", time_col=None, drop=[],
        desc=("HTRU2 pulsar candidates: 8 physical statistics of the "
              "integrated profile and DM-SNR curve. No cross-row entity, but "
              "every column carries physical meaning."),
    ),
    "adult": dict(
        name="adult", table="raw", label="income_gt_50k", time_col=None,
        drop=[],
        desc=("Adult Census Income, data+test concatenated, 48,842 rows. "
              "fnlwgt dropped as a sampling weight. Method evaluation only."),
    ),
    "spambase": dict(
        name="spambase", table="raw", label="is_spam", time_col=None, drop=[],
        desc=("Spambase: 57 hand-built word/character-frequency features. "
              "Already engineered, so it tests a second engineering layer."),
    ),
}
DATASET_ORDER = [
    "credit_default", "twitterbot", "fraudecom", "creditcard",
    "vehicleloan", "ieeecis", "sparknov", "sparknov_trainonly",
    "sparknov_external", "paysim", "talkingdata",
    "bank_marketing", "online_shoppers", "magic_gamma", "drybean",
    "occupancy",
    "htru2", "adult", "spambase",
]

MECHANISM_SUBSET = ["credit_default", "creditcard", "ieeecis", "paysim"]


def get_config(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {', '.join(DATASET_ORDER)}")
    cfg = dict(DATASETS[name])
    cfg["db_path"] = DB_DIR / f"{name}.duckdb"
    cfg.setdefault("table", "raw")
    cfg.setdefault("outer_test_fraction", 0.20)
    cfg.setdefault("search_val_fraction_of_outer", 0.20)
    cfg.setdefault("split_seed", 42)
    return cfg
