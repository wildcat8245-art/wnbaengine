"""Walk-forward backtest with a falsification test.

No historical real prop lines exist anywhere, so the backtest uses a
synthetic but defensible line: the player's own trailing rolling average for that stat,
at standard -110/-110 vig (decimal 1.909) on both sides. This tests three
real things against real held-out outcomes, side by side:

1. **parametric** -- the fitted Poisson/NegBinom/quantile model's own
   probability. Calibrated and profitable against this line?
2. **empirical_mc** -- the SAME engine that actually drives live picks in
   predict_props.py (empirical_resample_prob_over): bootstrap resampling
   from the player's own last MC_POOL_GAMES real games strictly BEFORE
   the game being bet on (no lookahead), falling back to the parametric
   probability when a player doesn't have enough real prior games yet --
   identical fallback rule to the live board. Until this was added, the
   backtest only validated the parametric model, not the mechanism that
   actually decides live bets.
3. **falsification** -- does the SAME staking engine, fed a naive 0.5 (no
   real model skill) instead of either real probability, produce ~flat
   bankroll? If a bankroll only beats flat because of the staking
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
from src.models.predict_props import (
    KELLY_FRACTION, MAX_STAKE_FRACTION, kelly_stake,
    empirical_resample_prob_over, MC_POOL_GAMES, MC_MIN_POOL_GAMES,
)

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


def build_player_histories(df: pd.DataFrame, target: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Precompute each player's full chronological (dates, values) arrays for
    a target, sorted, so per-row pool lookups below are a searchsorted (fast)
    instead of rescanning the whole dataframe once per test row."""
    histories: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for player, g in df.sort_values("game_date").groupby("player_name"):
        histories[player] = (g["game_date"].to_numpy(), g[target].to_numpy(dtype=float))
    return histories


def pool_before(
    histories: dict[str, tuple[np.ndarray, np.ndarray]],
    player: str,
    before_date,
    n_recent: int = MC_POOL_GAMES,
) -> np.ndarray:
    """The player's last n_recent real outcomes strictly before before_date --
    the same resampling universe build_player_game_pool uses live, generalized
    to "as of a given date" so a walk-forward backtest never lets a game peek
    at its own future (or the model's future) games."""
    dates, values = histories.get(player, (np.empty(0), np.empty(0)))
    idx = np.searchsorted(dates, np.datetime64(before_date), side="left")
    return values[max(0, idx - n_recent):idx]


def run_bankroll_sim(
    test: pd.DataFrame,
    target: str,
    models,
    histories: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Walks the test set forward in real chronological order, computing all
    three probability sources (parametric / empirical_mc / falsification) per
    row and staking each against its own independent bankroll, so all three
    strategies are compared on identical bets under identical conditions."""
    test = test.sort_values("game_date").reset_index(drop=True)
    bankrolls = {"parametric": STARTING_BANKROLL, "empirical_mc": STARTING_BANKROLL, "falsification": STARTING_BANKROLL}
    mc_fallback_count = 0
    history = []

    for _, row in test.iterrows():
        line = row[f"{target}_synthetic_line"]
        actual = row[target]
        x_row = row[FEATURE_COLS].to_frame().T

        if hasattr(models, "predict_prob_over"):
            parametric_prob = models.predict_prob_over(x_row, line)
        else:
            parametric_prob = predict_prob_over_row(models, x_row, line)

        # Same fallback rule as the live board: bootstrap only when there's
        # enough real prior history to do it meaningfully.
        pool = pool_before(histories, row["player_name"], row["game_date"])
        if len(pool) >= MC_MIN_POOL_GAMES:
            empirical_prob = empirical_resample_prob_over(pool, line)
        else:
            empirical_prob = parametric_prob
            mc_fallback_count += 1

        probs = {"parametric": parametric_prob, "empirical_mc": empirical_prob, "falsification": 0.5}

        record = {"game_date": row["game_date"]}
        for name, prob in probs.items():
            ev = prob * VIG_DECIMAL_ODDS
            stake = kelly_stake(prob, VIG_DECIMAL_ODDS, bankrolls[name]) if ev > EV_THRESHOLD else 0.0
            if stake > 0:
                won = actual > line
                bankrolls[name] += stake * (VIG_DECIMAL_ODDS - 1) if won else -stake
            record[f"{name}_bankroll"] = bankrolls[name]
            record[f"{name}_staked"] = stake
        history.append(record)

    result = pd.DataFrame(history)
    result.attrs["mc_fallback_count"] = mc_fallback_count
    result.attrs["mc_fallback_total"] = len(test)
    return result


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

        # Pool built from the FULL dataset (train+test), not just test --
        # pool_before() only ever looks strictly before each bet's own game
        # date, so an earlier real game (even one that's also technically
        # "in the test season") is genuine information a live bettor would
        # actually have had that day. This mirrors predict_props.py exactly.
        histories = build_player_histories(df, target)
        run = run_bankroll_sim(test, target, models, histories)

        fallback_n = run.attrs["mc_fallback_count"]
        fallback_total = run.attrs["mc_fallback_total"]
        print(f"  empirical_mc fell back to the parametric probability on "
              f"{fallback_n}/{fallback_total} bets ({fallback_n / fallback_total:.1%}) "
              f"-- player had fewer than {MC_MIN_POOL_GAMES} real prior games.")

        for name, label in [
            ("parametric", "Parametric model-driven"),
            ("empirical_mc", "Empirical Monte Carlo-driven (live board's real engine)"),
            ("falsification", "Falsification (p=0.5 always)"),
        ]:
            print(f"  {label}: final bankroll {run[f'{name}_bankroll'].iloc[-1]:.2f}, "
                  f"bets placed: {(run[f'{name}_staked'] > 0).sum()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
