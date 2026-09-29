# FORGE: Feature Optimization via Recursive Generation and Evaluation

Reference implementation of **FORGE**, a statistical harness for LLM-based feature engineering on
imbalanced tabular risk data. An LLM proposes candidate features; FORGE screens, admits and validates
them before they reach an unchanged downstream learner (LightGBM, XGBoost or CatBoost).

## Paper

Cheng-Yu Lin and Jyh-Shing Roger Jang, *FORGE: Feature Optimization via Recursive Generation and
Evaluation*, Department of Computer Science and Information Engineering, National Taiwan University, 2026.

```bibtex
@unpublished{lin2026forge,
  title  = {{FORGE}: Feature Optimization via Recursive Generation and Evaluation},
  author = {Lin, Cheng-Yu and Jang, Jyh-Shing Roger},
  year   = {2026},
  note   = {Department of Computer Science and Information Engineering, National Taiwan University}
}
```

The paper's results are in its Tables 5-9.

## Installation

Python 3.11, CPU only (if `python3.11` is missing, `uv python install 3.11` provides it).

```bash
git clone https://github.com/ntuaha/FORGE.git
cd FORGE
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

LightGBM needs OpenMP (`libgomp1` on Debian/Ubuntu, `brew install libomp` on macOS).

## Usage

Reproduce the paper's Credit Default / LightGBM / seed-11 result (no LLM needed, about 20 s):

```bash
python main.py
```

```text
configuration                          test AP       paper     |diff|
Raw baseline (this code)    0.5482592133059769           -          -
FORGE (Raw+CoF)             0.5502016247535624   0.5502016    2.5e-08
FORGE (Raw+CoF+Experts)     0.5502016247535624           -          -
OK: the published FORGE test AP is reproduced.
```

Other learners, seeds and datasets (compare with Tables 5-9 of the paper):

```bash
python main.py --seed 12                    # or set "seed: 12" in your own YAML file (see Settings)
python main.py --learner xgboost            # or catboost
python main.py --dataset temporal_demo --learner catboost   # small synthetic data that also runs the CoF rounds
python main.py --dataset sparknov           # needs the Kaggle files, see "Datasets"
python main.py --output outputs/run         # also write report.json, metrics.csv, features.json, predictions.npz
python -m unittest discover -s tests        # tests, including the end-to-end reproduction
```

**Settings.** Every parameter is in [`configs/default.yaml`](configs/default.yaml), with a comment naming
the paper section, equation or table it comes from; each dataset has a file in [`configs/datasets/`](configs/datasets/). Write
your own YAML with only the keys you want to change and run it; see
[`configs/example_experiment.yaml`](configs/example_experiment.yaml):

```bash
python main.py --config configs/example_experiment.yaml --output outputs/my_run   # saves the full settings as outputs/my_run/config.yaml
```

**Your own LLM.** Without an LLM, runs replay recorded proposals (shipped for Credit Default only). To let
an LLM propose features, use any command-line tool (the prompt goes to its stdin, the JSON answer is read
from its output) or an OpenAI-compatible API:

```bash
python main.py --dataset sparknov --llm-command "codex exec --skip-git-repo-check --output-schema {schema} -o {output} -"
python main.py --llm-command "claude -p --output-format text"
python main.py --llm-command "ollama run qwen3 --format json"

export FORGE_LLM_API_URL="https://<endpoint>/v1/chat/completions" FORGE_LLM_API_KEY="<key>"
python main.py --llm-mode api                # model: llm_model in configs/default.yaml (gpt-5.5)
```

Answers are cached in `.cache/llm/`, so rerunning with `--llm-mode replay` repeats a run exactly (use a
separate `cache_dir` per LLM when comparing LLMs). The tool must already be installed and logged in.
`--output DIR` also saves the complete settings of the run as `DIR/config.yaml`.

**Your own learner.** Besides `lightgbm`, `xgboost` and `catboost`, `--learner` accepts
`module:function`, where the function returns any classifier with `fit` and `predict_proba`; see
[`examples/my_learner.py`](examples/my_learner.py):

```bash
python main.py --learner examples.my_learner:make_model
```

**Your own data.** Copy [`configs/datasets/my_data_example.yaml`](configs/datasets/my_data_example.yaml),
set the CSV file, the 0/1 label, the event-time column (numeric or dates; `null` for a stratified split),
entity columns and columns to drop, and run `python main.py --config my_data.yaml`.

`notebooks/FORGE_walkthrough.ipynb` walks through every stage on Credit Default
(`pip install jupyter matplotlib`, then open it from the `notebooks/` folder).

## Datasets

The datasets are not redistributed. `main.py` downloads Credit Default. For the others, download the
files below into `data/<dataset>/` (paths are set in `configs/datasets/<dataset>.yaml`);
`src/forge/datasets.py` turns them into the tables used in the paper.

| Dataset | Files | Source |
|---|---|---|
| Credit Default | downloaded automatically | UCI, "Default of Credit Card Clients" (Yeh and Lien, 2009), <https://archive.ics.uci.edu/dataset/350> |
| Fraudecom | `fraudecom/Fraud_Data.csv` | Kaggle, "Fraud ecommerce", <https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce> |
| Vehicleloan | `vehicleloan/train.csv` | Kaggle, "L&T Vehicle Loan Default Prediction", <https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction> |
| Sparknov | `sparknov/fraudTrain.csv`, `sparknov/fraudTest.csv` | Kaggle, "Credit Card Transactions Fraud Detection Dataset", <https://www.kaggle.com/datasets/kartik2112/fraud-detection> |
| Banksim | `banksim/bs140513_032310.csv` | Kaggle, "BankSim" (Lopez-Rojas and Axelsson, 2014), <https://www.kaggle.com/datasets/ealaxi/banksim1> |
| Creditcard | `creditcard/creditcard.csv` | Kaggle, "Credit Card Fraud Detection" (Dal Pozzolo et al., 2015), <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> |
| Twitterbot | `twitterbot/twitter_human_bots_dataset.csv` | Kaggle, "Twitter Bots Accounts", <https://www.kaggle.com/datasets/davidmartngutirrez/twitter-bots-accounts> |
