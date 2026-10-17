"""Shared NFL loading, feature engineering, splitting, and preprocessing."""

import nflreadpy as nfl
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


# Columns for inspection and splits; season is also a candidate era predictor
ID_COLUMNS = ["season", "game_id", "play_id"]

# Current list of feature columns for modeling; includes engineered features
FEATURE_COLUMNS = [
    "ydstogo",
    "yardline_100",
    "goal_to_go",
    "qtr",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "score_differential",
    "posteam",
    "defteam",
    "posteam_type",
    "roof",
    "surface",
    "temp",
    "wind",
    "week",
    "season_type",
    "season",
    # Feature engineered by process_4th_down_data():
    "red_zone",
    "short_yardage",
    "end_of_half",
    "late_game_deficit",
    "overtime",
    "log_ydstogo",
    "distance_x_goal_to_go",
    "distance_x_red_zone",
    "timeout_advantage",
]
TARGET_COLUMN = "converted"

# Feature groups for scaling and encoding, based on meaning rather than dtype.
BINARY_COLUMNS = [
    "goal_to_go", "red_zone", "short_yardage", "end_of_half",
    "late_game_deficit", "overtime",
]
CATEGORICAL_COLUMNS = [
    "posteam", "defteam", "posteam_type", "roof", "surface", "season_type",
]
NUMERIC_COLUMNS = [
    column for column in FEATURE_COLUMNS
    if column not in BINARY_COLUMNS + CATEGORICAL_COLUMNS
]


def load_data(seasons: int | list[int]) -> pd.DataFrame:
    """Load explicit NFL seasons from nflverse as a pandas DataFrame.

    nflreadpy handles downloading and its own caching. The first call requires
    internet access. Pass a year or a nonempty list of years (1999 onward).
    Returns all plays and columns, leaving filtering to process_4th_down_data().
    """
    if isinstance(seasons, int) and not isinstance(seasons, bool):
        seasons = [seasons]
    if not isinstance(seasons, list) or not seasons:
        raise ValueError("Provide a season year or a nonempty list of years.")
    if any(type(year) is not int or year < 1999 for year in seasons):
        raise ValueError("Seasons must be integer years from 1999 onward.")

    raw = nfl.load_pbp(sorted(set(seasons)))
    return raw.to_pandas()


def process_4th_down_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Keep fourth-down run/pass attempts without penalties and with clear outcomes.

    Retains all columns; adds converted and the notebook's situation features:
    red_zone, short_yardage, end_of_half, late_game_deficit, overtime,
    log_ydstogo, distance interactions, and timeout_advantage.
    Indicators are nullable 0/1 integers; missing inputs remain missing.
    Returns a copy without scaling or encoding; select predictors separately.
    """
    # Check the columns needed for filtering, labeling, and feature engineering.
    required = [
        "down", "play_type", "penalty", "fourth_down_converted", "fourth_down_failed",
        "yardline_100", "ydstogo", "qtr", "half_seconds_remaining",
        "game_seconds_remaining", "score_differential", "goal_to_go",
        "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    ]
    missing = sorted(set(required) - set(raw_data.columns))
    if missing:
        raise ValueError(f"Missing required play-by-play columns: {', '.join(missing)}")

    # Keep outcomes where exactly one of the success/failure flags is 1.
    success = raw_data["fourth_down_converted"]
    failure = raw_data["fourth_down_failed"]
    valid_outcome = success.isin([0, 1]) & failure.isin([0, 1]) & (success + failure).eq(1)
    # Select fourth-down runs and passes, excluding all flagged penalty plays.
    attempts = (
        raw_data["down"].eq(4)
        & raw_data["play_type"].isin(["run", "pass"])
        & raw_data["penalty"].eq(0)
        & valid_outcome
    )
    # Keep every column without modifying the original data.
    processed = raw_data.loc[attempts].copy()
    # Use the success flag as the binary target, replacing any existing label.
    processed[TARGET_COLUMN] = success.loc[attempts].astype("int8")

    # FEATURE ENGINEERING
    # Nullable numeric values preserve missing inputs in derived features.
    situation = processed[[
        "yardline_100", "ydstogo", "qtr", "half_seconds_remaining",
        "game_seconds_remaining", "score_differential", "goal_to_go",
        "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    ]].astype("Float64")

    # Indicators for red zone and short yardage
    processed["red_zone"] = situation["yardline_100"].between(0, 20).astype("Int64")
    processed["short_yardage"] = situation["ydstogo"].between(0, 2).astype("Int64")

    # Indicator for final two minutes of a regulation half
    processed["end_of_half"] = (
        (situation["qtr"].eq(2) | situation["qtr"].eq(4))
        & situation["half_seconds_remaining"].between(0, 120)
    ).astype("Int64")

    # Indicator for trailing in the final five minutes of the fourth quarter
    processed["late_game_deficit"] = (
        situation["qtr"].eq(4)
        & situation["game_seconds_remaining"].between(0, 300)
        & situation["score_differential"].lt(0)
    ).astype("Int64")
    processed["overtime"] = situation["qtr"].ge(5).astype("Int64")

    # Nonlinear distance, interactions, and relative timeout availability
    processed["log_ydstogo"] = np.log1p(situation["ydstogo"])
    processed["distance_x_goal_to_go"] = situation["ydstogo"] * situation["goal_to_go"]
    processed["distance_x_red_zone"] = situation["ydstogo"] * processed["red_zone"]
    processed["timeout_advantage"] = (
        situation["posteam_timeouts_remaining"] - situation["defteam_timeouts_remaining"]
    )

    # Cleaning surfaces so "grass" does not appear twice
    processed["surface"] = (
    processed["surface"]
    .astype("string")
    .str.strip()
    .replace("", pd.NA)
    )

    return processed.reset_index(drop=True)


def split_data(
    data: pd.DataFrame, random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, test, validation) with approximate 60/20/20 row counts.

    Stratifies by season in both split steps. The seed reproduces splits for
    identical input row order. Keeps all columns and original indices; returns
    independent copies. Each season needs enough rows for both stratified splits.
    This is a random play-level split, so plays from one game can cross partitions.
    """
    if "season" not in data.columns:
        raise ValueError("A season column is required to stratify the splits.")
    if data.empty or data["season"].isna().any():
        raise ValueError("Provide nonempty data with no missing season values.")

    # Reserve 40% for the equally sized test and validation partitions.
    train, held_out = train_test_split(
        data, test_size=0.4, random_state=random_seed, stratify=data["season"],
    )
    test, validation = train_test_split(
        held_out, test_size=0.5, random_state=random_seed,
        stratify=held_out["season"],
    )
    return train.copy(), test.copy(), validation.copy()


def _as_float(values):
    """Keep imputed binary columns numeric even when their input dtype is object."""
    return values.astype(float)


def make_preprocessor(*, scale_numeric: bool = True) -> ColumnTransformer:
    """Build an unfitted preprocessor; fit only on the training partition.

    Numeric: median imputation and optional standardization (mean 0, variance 1).
    Binary: most-frequent imputation, retaining 0/1 values without scaling.
    Categorical: a missing category and one-hot encoding; unseen categories
    produce all-zero indicators for that feature. Returns a dense DataFrame.
    Disable scaling for tree models. Extra input columns, including targets,
    are dropped; all named input features must be present.
    """
    numeric_steps = [
        ("impute", SimpleImputer(
            missing_values=pd.NA, strategy="median", keep_empty_features=True,
        )),
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical = Pipeline([
        ("impute", SimpleImputer(
            missing_values=pd.NA, strategy="constant", fill_value="missing",
            keep_empty_features=True,
        )),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC_COLUMNS),
        ("binary", Pipeline([
            ("impute", SimpleImputer(
                missing_values=pd.NA, strategy="most_frequent", keep_empty_features=True,
            )),
            ("as_float", FunctionTransformer(_as_float, feature_names_out="one-to-one")),
        ]), BINARY_COLUMNS),
        ("categorical", categorical, CATEGORICAL_COLUMNS),
    ], remainder="drop").set_output(transform="pandas")


def split_data_chronological(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split fourth-down plays chronologically by NFL season.

    Training: 2015–2022
    Validation: 2023
    Test: 2024–2025

    Returns (train, test, validation), matching split_data().
    """
    train = data[data["season"].between(2015, 2022)].copy()
    validation = data[data["season"] == 2023].copy()
    test = data[data["season"].between(2024, 2025)].copy()

    return train, test, validation

