"""Walk-forward backtest with a falsification test.

No historical real prop lines exist (same limitation the paper itself
flags for its own player-prop test), so the backtest uses a synthetic but
defensible line: the player's own trailing rolling average for that stat,
at standard -110/-110 vig (decimal 1.909) on both sides. This tests two
real things against real held-out outcomes:

1. Is the model calibrated and profitable against this line, under
   fractional-Kelly + EV-threshold staking?
2. Falsification: does the SAME staking engine, fed a naive 0.5 (no real
   model skill) instead of the model's probability, produce ~flat bankroll?
   If the model's bankroll only beats flat because of the staking
   mechanism itself (not real skill), this test catches it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.train_baseline_model import (
    FEATURE_COLS, TARGETS, POISSON_TARGETS, NEGATIVE_BINOMIAL_TARGETS,
    load_dataset, train_quantile_models, train_poisson_model,
    train_negative_binomial_model, train_test_split_by_season,
)
from src.models.predict_props import KELLY_FRACTION, MAX_STAKE_FRACTION, kelly_stake

VIG_DECIMAL_ODDS = 1.909  # standard -110
EV_THRESHOLD = 1.02
STARTING_BANKROLL = 1000.0


def predict_prob_over_row(models: dict[float, object], x_row: pd.DataFrame, line: float) -> float:
    qs = sorted(models.keys())
    preds = sorted(model.predict(x_row)[0] for model in models.values())
    preds = np.maximum.accumulate(preds)
    if line <= preds[0]:
        prob = 1 - qs[0] / 2
    elif line >= preds[-1]:
        prob = (1 - qs[-1]) / 2
    else:
        cdf_at_line = np.interp(line, preds, qs)
        prob = 1 - cdf_at_line
    # No prop model should claim near-certainty; clamp to a believable range.
    return float(np.clip(prob, 0.05, 0.95))


def run_bankroll_sim(test: pd.DataFrame, target: str, models, use_model_prob: bool) -> pd.DataFrame:
    test = test.sort_values("game_date").reset_index(drop=True)
    bankroll = STARTING_BANKROLL
    history = []

    for _, row in test.iterrows():
        line = row[f"{target}_synthetic_line"]
        actual = row[target]
        x_row = row[FEATURE_COLS].to_frame().T

        if use_model_prob:
            if hasattr(models, "predict_prob_over"):
                prob_over = models.predict_prob_over(x_row, line)
            else:
                prob_over = predict_prob_over_row(models, x_row, line)
        else:
            prob_over = 0.5  # falsification: no real skill, just the staking mechanism

        ev = prob_over * VIG_DECIMAL_ODDS
        stake = kelly_stake(prob_over, VIG_DECIMAL_ODDS, bankroll) if ev > EV_THRESHOLD else 0.0

        if stake > 0:
            won = actual > line
            bankroll += stake * (VIG_DECIMAL_ODDS - 1) if won else -stake

        history.append({"game_date": row["game_date"], "bankroll": bankroll, "staked": stake})

    return pd.DataFrame(history)


def main() -> int:
    df = load_dataset(Path("data/processed/player_features.csv"))
    train, test = train_test_split_by_season(df, test_seasons={2025, 2026})
    print(f"Train: {len(train)} rows, Test: {len(test)} rows")

    # Synthetic line: player's trailing rolling average for the raw stat,
    # rounded to the nearest 0.5 (mimics how books actually set lines).
    for target in TARGETS:
        test = test.copy()
        raw_estimate = test[f"{target}_per100_last5"] / 100 * test["possessions"]
        # Round to the NEAREST 0.5 (unbiased) -- floor()+0.5 was still
        # systematically below the estimate whenever the fractional part
        # exceeded 0.5, which is what caused the runaway bankroll below.
        test[f"{target}_synthetic_line"] = np.round(raw_estimate * 2) / 2

    for target in TARGETS:
        print(f"\n=== {target} ===")
        if target in POISSON_TARGETS:
            models = train_poisson_model(train, target)
            print("  using Poisson model (discrete count target)")
        elif target in NEGATIVE_BINOMIAL_TARGETS:
            models = train_negative_binomial_model(train, target)
            print("  using Negative Binomial model (overdispersed count target)")
        else:
            models = train_quantile_models(train, target)

        real_run = run_bankroll_sim(test, target, models, use_model_prob=True)
        falsified_run = run_bankroll_sim(test, target, models, use_model_prob=False)

        print(f"Model-driven:  final bankroll {real_run['bankroll'].iloc[-1]:.2f}, "
              f"bets placed: {(real_run['staked'] > 0).sum()}")
        print(f"Falsification (p=0.5 always): final bankroll {falsified_run['bankroll'].iloc[-1]:.2f}, "
              f"bets placed: {(falsified_run['staked'] > 0).sum()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
