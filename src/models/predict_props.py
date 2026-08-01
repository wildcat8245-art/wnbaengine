"""Tie it together: real live odds + calibrated model -> a real prop board.

For each real prop line fetched today, predicts P(stat > line) from the
player's latest feature row via interpolated quantile CDF, computes EV
against the real decimal odds, and sizes a fractional-Kelly stake.

Usage:
    python -m src.models.predict_props
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.train_baseline_model import (
    FEATURE_COLS, POISSON_TARGETS, NEGATIVE_BINOMIAL_TARGETS,
    load_dataset, train_poisson_model, train_negative_binomial_model, train_quantile_models,
)

MARKET_TO_TARGET = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
}

KELLY_FRACTION = 0.3
EV_THRESHOLD = 1.05
MAX_STAKE_FRACTION = 0.05
BANKROLL = 1000.0


def predict_prob_over(models: dict[float, object], x_row: pd.DataFrame, line: float) -> float:
    """Interpolate the fitted quantile curve to estimate P(stat > line)."""
    qs = sorted(models.keys())
    preds = sorted(model.predict(x_row)[0] for model in models.values())
    # Enforce monotonicity (quantile crossing can happen with independently fit models).
    preds = np.maximum.accumulate(preds)

    if line <= preds[0]:
        return 1 - qs[0] / 2  # line below our lowest modeled quantile: treat as very likely over
    if line >= preds[-1]:
        return (1 - qs[-1]) / 2  # line above our highest modeled quantile: treat as very unlikely over

    cdf_at_line = np.interp(line, preds, qs)
    return 1 - cdf_at_line


def kelly_stake(prob: float, decimal_odds: float, bankroll: float) -> float:
    b = decimal_odds - 1
    q = 1 - prob
    f = (b * prob - q) / b
    f = max(f, 0.0) * KELLY_FRACTION
    f = min(f, MAX_STAKE_FRACTION)
    return f * bankroll


def latest_features_by_player(features: pd.DataFrame) -> pd.DataFrame:
    features = features.sort_values("game_date")
    return features.groupby("player_name").tail(1).set_index("player_name")


# Trend-conflict guard: catches exactly the Rebecca Allen case (real recent
# rate + minutes both trending up, but the model recommends Under anyway).
# Requires the trend to show up in BOTH the per-100 rate AND minutes played,
# not just one noisy stat, before flagging -- a single-game outlier in rate
# alone shouldn't trigger this. Thresholds are deliberately blunt (25%/15%)
# so it only fires on a real, visible trend, not everyday sampling noise.
TREND_RATE_THRESHOLD = 1.25
TREND_MINUTES_THRESHOLD = 1.15


def trend_conflict_flag(side: str, last5: float, last10: float, min5: float, min10: float) -> str:
    rate_ratio = last5 / max(last10, 0.1)
    minutes_ratio = min5 / max(min10, 1.0)

    trending_up = rate_ratio >= TREND_RATE_THRESHOLD and minutes_ratio >= TREND_MINUTES_THRESHOLD
    trending_down = rate_ratio <= 1 / TREND_RATE_THRESHOLD and minutes_ratio <= 1 / TREND_MINUTES_THRESHOLD

    if side == "Under" and trending_up:
        return "CONFLICT: trending UP, model says Under -- STAY AWAY"
    if side == "Over" and trending_down:
        return "CONFLICT: trending DOWN, model says Over -- STAY AWAY"
    return ""


def main() -> int:
    features = load_dataset(Path("data/processed/player_features.csv"))
    latest = latest_features_by_player(features)

    prop_files = sorted(glob.glob("data/raw/daily_props/props_*.csv"))
    if not prop_files:
        print("No daily props snapshot found. Run fetch_daily_props first.")
        return 1
    props = pd.read_csv(prop_files[-1])
    print(f"Using {prop_files[-1]}: {len(props)} prop rows")

    models_by_target = {}
    for target in set(MARKET_TO_TARGET.values()):
        print(f"Training final {target} model on all available history...")
        if target in POISSON_TARGETS:
            models_by_target[target] = train_poisson_model(features, target)
        elif target in NEGATIVE_BINOMIAL_TARGETS:
            models_by_target[target] = train_negative_binomial_model(features, target)
        else:
            models_by_target[target] = train_quantile_models(features, target)

    # One prob_over per (player, market, line) -- doesn't depend on side/book.
    # Odds DO depend on side and book, so both sides get evaluated against
    # their own real quoted odds and the better side is reported.
    results = []
    grouped = props[props["market"].isin(MARKET_TO_TARGET)].groupby(
        ["player_name", "market", "line", "bookmaker"]
    )
    for (player, market, line, book), group in grouped:
        target = MARKET_TO_TARGET[market]
        if player not in latest.index:
            continue

        over_row = group[group["side"] == "Over"]
        under_row = group[group["side"] == "Under"]
        if over_row.empty or under_row.empty:
            continue  # need both sides' real odds to evaluate fairly

        x_row = latest.loc[[player], FEATURE_COLS]
        if x_row.isna().any(axis=None):
            continue

        model = models_by_target[target]
        if hasattr(model, "predict_prob_over"):
            prob_over = model.predict_prob_over(x_row, line)
        else:
            prob_over = predict_prob_over(model, x_row, line)

        over_odds = over_row["decimal_odds"].iloc[0]
        under_odds = under_row["decimal_odds"].iloc[0]
        ev_over = prob_over * over_odds
        ev_under = (1 - prob_over) * under_odds

        if ev_over >= ev_under:
            side, prob, odds, ev = "Over", prob_over, over_odds, ev_over
        else:
            side, prob, odds, ev = "Under", 1 - prob_over, under_odds, ev_under

        last5 = x_row[f"{target}_per100_last5"].iloc[0]
        last10 = x_row[f"{target}_per100_last10"].iloc[0]
        min5 = x_row["minutes_last5"].iloc[0]
        min10 = x_row["minutes_last10"].iloc[0]

        flag = trend_conflict_flag(side, last5, last10, min5, min10)
        # A flagged pick is a stay-away by definition -- no stake regardless
        # of what the raw EV says, since the whole point of the flag is that
        # the model and the player's own recent trend actively disagree.
        stake = 0.0 if flag else (kelly_stake(prob, odds, BANKROLL) if ev > EV_THRESHOLD else 0.0)

        results.append({
            "player": player,
            "market": target,
            "side": side,
            "line": line,
            "book": book,
            "decimal_odds": odds,
            "model_prob": round(prob, 3),
            "ev": round(ev, 3),
            "kelly_stake": round(stake, 2),
            "flag": flag,
            # Supporting context -- shown for every pick, not just when asked.
            # Generic column names (not f"{target}_...") so every row uses the
            # same columns regardless of market -- otherwise the board would
            # end up sparse/inconsistent across points/rebounds/assists rows.
            "stat_per100_last5": round(last5, 2),
            "stat_per100_last10": round(last10, 2),
            "minutes_last5_vs_last10": f"{min5:.1f} vs {min10:.1f}",
        })

    board = pd.DataFrame(results).sort_values("ev", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    print(board.to_string(index=False))

    flagged = board[board["flag"] != ""]
    positive_ev = board[board["kelly_stake"] > 0]
    print(f"\n{len(positive_ev)} bets clear the EV>{EV_THRESHOLD} threshold out of {len(board)} evaluated.")
    if not flagged.empty:
        print(f"\n{len(flagged)} pick(s) had a high EV but were caught by the trend-conflict guard "
              f"(stake forced to 0, not recommended):")
        print(flagged[["player", "market", "side", "line", "ev", "stat_per100_last5",
                        "stat_per100_last10", "minutes_last5_vs_last10"]].drop_duplicates(
                            subset=["player", "market", "side", "line"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
