# Predicting 4th Down Conversion Probability in the NFL

A DATASCI 207 project exploring how machine learning can help inform NFL fourth-down decisions.

We plan to predict the probability of a successful fourth-down conversion using information available before the snap, such as yards to go, field position, time remaining, score, and team information. We will also explore how these predictions can inform decisions that maximize win probability.

## Data

NFL play-by-play data from nflverse, accessed through the Python package `nflreadpy`. Our analysis will focus on fourth-down plays where the offense attempted to convert.

## Team

- Logan Calder
- Jordan Rivera
- Emily Silvershein
- Ariel Gholar

## Proposal

See the [project proposal](Project%20Proposal.pdf) for more details.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from the repository folder:

```bash
uv sync
uv run jupyter lab
```

The project uses Python 3.12. `uv sync` installs the Python version and dependencies as needed into a local `.venv` environment.

In VS Code, select the project's `.venv` as the notebook kernel.

Shared code lives in the root-level `nfl_project/` package. `uv sync` installs it in editable mode, so changes to its Python files are available without reinstalling. Restart the notebook kernel or reload the module to pick up changes to code already imported.

## Loading data

From a notebook using the project environment:

```python
from nfl_project.data import FEATURE_COLUMNS, load_data, process_4th_down_data

raw_data = load_data(seasons=[2022, 2023, 2024])
attempts = process_4th_down_data(raw_data)
X = attempts[FEATURE_COLUMNS]  # EDA candidates; refine through model comparisons
y = attempts["converted"]
```

`process_4th_down_data` keeps fourth-down run/pass attempts without penalties and with clear outcomes. It retains original columns and adds the `converted` target plus engineered features. Use `FEATURE_COLUMNS` to select pre-snap model inputs.

## Splitting and preprocessing

`split_data` uses a season-stratified 60/20/20 train/test/validation split with seed 42 by default. Plays from the same game may appear in different splits.

`make_preprocessor` imputes missing values, standardizes numeric features, leaves binary indicators as 0/1, and one-hot encodes categories. Fit it only on training data.

```python
from nfl_project.data import make_preprocessor, split_data

train, test, validation = split_data(attempts)
preprocessor = make_preprocessor()
X_train = preprocessor.fit_transform(train[FEATURE_COLUMNS])
X_test = preprocessor.transform(test[FEATURE_COLUMNS])
X_valid = preprocessor.transform(validation[FEATURE_COLUMNS])
```

Use `make_preprocessor(scale_numeric=False)` for tree models.

Field definitions: [nflverse play-by-play dictionary](https://nflreadr.nflverse.com/articles/dictionary_pbp.html).

Work in progress.

## Logan's EDA

`notebooks/02_logan_eda.ipynb` is owned by Logan Calder. It reuses the shared data pipeline, audits filtering and training-feature quality, and exports four labeled figures plus summary tables and milestone evidence to `reports/logan_eda/`.

Run all cells in order from the project root or `notebooks/` using the project environment. Cells have explicit numbering beginning at 1. The first run downloads 2015–2025 play-by-play data and caches it under the ignored `data/logan_eda/` folder. The cache metadata records the source download time; remove the cached parquet explicitly if a fresh source snapshot is desired.

The notebook proposes training on 2015–2022, validating on 2023, and testing on 2024–2025, pending team agreement. Feature/target exploration uses only the proposed training partition. The exported `run_metadata.json` records the package versions used for the saved results. No model has been fitted in this notebook.
