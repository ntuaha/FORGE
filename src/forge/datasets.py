"""Dataset loaders.

``cfg['loader']`` (set in ``configs/datasets/<name>.yaml``) selects a function
below. Each ``prepare_*`` function turns the Kaggle files of one paper dataset
into the table used in the experiments (Table 1):
column names are lower-cased, identifiers and free-text fields removed, and
timestamps converted to numbers. The files are read from ``data_dir`` and
are not redistributed.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .data import validate_dataset

UCI_CREDIT_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
CREDIT_LABEL = "default.payment.next.month"


# --------------------------------------------------------------------------- Credit Default (UCI)
def download_credit_default(data_dir: Path) -> Path:
    """Download the official UCI archive once into ``data_dir`` and return its path."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "credit_default_uci.zip"
    if not target.exists():
        with urllib.request.urlopen(UCI_CREDIT_URL, timeout=120) as response:
            payload = response.read()
        tmp = target.with_suffix(".part")
        tmp.write_bytes(payload)
        tmp.replace(target)
    return target


def read_credit_default(path: Path) -> pd.DataFrame:
    """Read Credit Default from the official ZIP/XLS or a CSV copy (e.g. Kaggle's UCI_Credit_Card.csv).

    Column names are lower-cased; the label is ``default.payment.next.month``.
    """
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.lower().endswith(".xls"))
            frame = pd.read_excel(io.BytesIO(archive.read(name)), header=1)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        frame = pd.read_excel(path, header=1)
    else:
        frame = pd.read_csv(path)
    frame.columns = [str(c).strip().lower() for c in frame]
    frame = frame.rename(columns={"default payment next month": CREDIT_LABEL})
    frame[CREDIT_LABEL] = pd.to_numeric(frame[CREDIT_LABEL], errors="raise").astype("int8")
    if frame.shape != (30000, 25) or int(frame[CREDIT_LABEL].sum()) != 6636:
        raise ValueError("expected the full UCI Credit Default data: 30,000 x 25 with 6,636 positives")
    return frame


def load_credit_default(cfg: dict) -> pd.DataFrame:
    files = _files(cfg, required=False)
    return read_credit_default(files[0] if files else download_credit_default(cfg["data_dir"]))


# --------------------------------------------------------------------------- Kaggle datasets
def _files(cfg: dict, required: bool = True) -> list[Path]:
    paths = [p if Path(p).is_absolute() else Path(cfg["data_dir"]) / p for p in cfg["data_files"]]
    missing = [str(p) for p in paths if not Path(p).exists()]
    if required and (missing or not paths):
        raise FileNotFoundError(f"dataset {cfg['dataset']!r} needs {missing or 'data_files'}; download it from the "
                                "source listed in README.md (the data is not redistributed)")
    return [Path(p) for p in paths]


def _read(cfg: dict, **kwargs) -> pd.DataFrame:
    frame = pd.concat([pd.read_csv(p, **kwargs) for p in _files(cfg)], ignore_index=True)
    frame.columns = ["".join(ch if ch.isalnum() else "_" for ch in str(c).strip().lower()) for c in frame]
    return frame


def _epoch_seconds(values: pd.Series, **kwargs) -> pd.Series:
    return (pd.to_datetime(values, **kwargs) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)


def prepare_fraudecom(cfg: dict) -> pd.DataFrame:
    """Fraud_Data.csv: signup and purchase times as epoch seconds (11 columns)."""
    frame = _read(cfg)
    frame["signup_ts"] = _epoch_seconds(frame.pop("signup_time"))
    frame["purchase_ts"] = _epoch_seconds(frame.pop("purchase_time"))
    return frame


def prepare_vehicleloan(cfg: dict) -> pd.DataFrame:
    """train.csv: dd-mm-yyyy dates parsed (41 columns; disbursal date is the event time)."""
    frame = _read(cfg).rename(columns={"disbursaldate": "disbursal_date"})
    for column in ("date_of_birth", "disbursal_date"):
        frame[column] = pd.to_datetime(frame[column], dayfirst=True)
    return frame


def prepare_sparknov(cfg: dict) -> pd.DataFrame:
    """fraudTrain.csv + fraudTest.csv; names, address, transaction id and row index removed,
    date of birth reduced to the year (17 columns)."""
    frame = _read(cfg)
    frame["dob_year"] = pd.to_datetime(frame["dob"]).dt.year
    return frame.drop(columns=["unnamed__0", "trans_date_trans_time", "first", "last", "street", "dob",
                               "trans_num"], errors="ignore")


def prepare_banksim(cfg: dict) -> pd.DataFrame:
    """bs140513_032310.csv: the quotes around every string value are removed (10 columns)."""
    frame = _read(cfg)
    for column in frame.select_dtypes(exclude="number"):
        frame[column] = frame[column].str.strip("'")
    for column in ("zipcodeori", "zipmerchant"):
        frame[column] = pd.to_numeric(frame[column])
    return frame


def prepare_creditcard(cfg: dict) -> pd.DataFrame:
    """creditcard.csv as is: Time, V1-V28, Amount, Class (31 columns)."""
    return _read(cfg)


def prepare_twitterbot(cfg: dict) -> pd.DataFrame:
    """twitter_human_bots_dataset.csv: is_bot = (account_type == "bot"), creation time as epoch
    seconds; text, URL and image fields removed (15 columns)."""
    frame = _read(cfg)
    frame["is_bot"] = (frame.pop("account_type") == "bot").astype(int)
    frame["created_ts"] = _epoch_seconds(frame.pop("created_at"))
    keep = ["default_profile", "default_profile_image", "favourites_count", "followers_count", "friends_count",
            "geo_enabled", "id", "lang", "location", "statuses_count", "verified", "average_tweets_per_day",
            "account_age_days", "is_bot", "created_ts"]
    return frame[keep]


def load_csv(cfg: dict) -> pd.DataFrame:
    """Your own data: the CSV files in ``data_files``, concatenated, column names unchanged."""
    return pd.concat([pd.read_csv(p) for p in _files(cfg)], ignore_index=True)


# --------------------------------------------------------------------------- synthetic demo
def make_temporal_demo(cfg: dict | None = None) -> pd.DataFrame:
    """Synthetic time-ordered teaching data (8,000 rows, fixed generator seed).

    The event rate depends on a debt/income burden and on an ``x1*x2``
    interaction, so the CoF rounds have something to find. It is not a paper
    benchmark.
    """
    rng = np.random.default_rng(20260831)
    n = 8000
    income, debt = rng.uniform(1, 10, n), rng.uniform(0, 12, n)
    x1, x2 = rng.normal(size=(2, n))
    region = rng.choice(["north", "south", "east"], n)
    probability = expit(-2.0 + 1.2 * debt / income + 1.4 * x1 * x2 + 0.3 * (region == "east"))
    return pd.DataFrame({"event_time": np.arange(n), "customer": np.arange(n) % 200,
                         "income": income, "debt": debt, "x1": x1, "x2": x2,
                         "region": region, "target": rng.binomial(1, probability)})


LOADERS = {"credit_default": load_credit_default, "temporal_demo": make_temporal_demo, "csv": load_csv,
           "fraudecom": prepare_fraudecom, "vehicleloan": prepare_vehicleloan, "sparknov": prepare_sparknov,
           "banksim": prepare_banksim, "creditcard": prepare_creditcard, "twitterbot": prepare_twitterbot}


def load_dataset(cfg: dict) -> pd.DataFrame:
    """Load ``cfg['dataset']`` with the loader named in ``cfg['loader']`` and validate it."""
    if cfg["loader"] not in LOADERS:
        raise ValueError(f"unknown loader {cfg['loader']!r}; pass the data with run_forge(cfg, frame=df) "
                         f"or use one of {sorted(LOADERS)}")
    return validate_dataset(LOADERS[cfg["loader"]](cfg), cfg)
