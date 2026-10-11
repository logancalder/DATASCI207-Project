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

Work in progress.
