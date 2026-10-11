"""Shared NFL data loading and fourth-down preparation for notebooks."""

import nflreadpy as nfl
import pandas as pd


# Identifiers are useful for inspection and train/test splits, not model inputs.
ID_COLUMNS = ["season", "game_id", "play_id"]
# Optional pre-snap baseline; process_4th_down_data retains all columns for exploration.
FEATURE_COLUMNS = [
    "ydstogo",
    "yardline_100",
    "qtr",
    "game_seconds_remaining",
    "score_differential",
    "posteam",
    "defteam",
]
TARGET_COLUMN = "converted"


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

    Retains all columns and adds converted (1 = success, 0 = failure).
    Returns a copy; select pre-snap features separately before training.
    """
    # Check that the columns needed for filtering and labeling are present.
    required = [
        "down", "play_type", "penalty", "fourth_down_converted", "fourth_down_failed"
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
    # Give the filtered rows a consecutive index starting at zero.
    return processed.reset_index(drop=True)
