"""Real backtest of a POOLED usage-vacuum effect (not one specific pair).

The per-pair on/off split (on_off_splits.py) failed validation twice --
correlation ~0 estimating one specific (player, teammate) pair's effect
from only 5-8 real games. Real, much larger-sample alternative: pool
across EVERY real instance where ANY normal rotation player missed a game,
and measure the average real effect on the REST of the roster's combined
per-36 rate, rather than trying to pin down which specific individual
benefits.

Same in-season validation discipline as the fixed on/off-split attempt:
fit on the first half of a real season, check whether it predicts the
second half of the SAME season.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.on_off_splits import load_played_minutes

ROTATION_MIN_MINUTES = 15.0  # a player averaging this or more is a "normal rotation player"
SEASONS = [2024, 2025, 2026]


def season_halves(df: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    season = df[df["game_date"].dt.year == year]
    if season.empty:
        return season, season
    midpoint = season["game_date"].quantile(0.5)
    return season[season["game_date"] < midpoint], season[season["game_date"] >= midpoint]


def pooled_effect(half: pd.DataFrame, team: str) -> dict | None:
    """Real pooled effect: for this team in this half-season, the rest of
    the roster's combined real per-36 rate in games missing >=1 normal
    rotation player vs. games at full strength."""
    team_rows = half[half["team"] == team]
    if team_rows.empty:
        return None

    per_player_avg_minutes = team_rows.groupby("player_name")["minutes"].mean()
    rotation = set(per_player_avg_minutes[per_player_avg_minutes >= ROTATION_MIN_MINUTES].index)
    if len(rotation) < 3:
        return None

    games = team_rows["game_id"].unique()
    game_rotation_present = {}
    for gid in games:
        game_rows = team_rows[team_rows["game_id"] == gid]
        played = set(game_rows.loc[game_rows["minutes"].fillna(0) > 0, "player_name"])
        game_rotation_present[gid] = rotation & played

    full_strength_games = {g for g, present in game_rotation_present.items() if len(present) == len(rotation)}
    missing_games = {g for g, present in game_rotation_present.items() if 0 < len(present) < len(rotation)}
    if len(full_strength_games) < 5 or len(missing_games) < 5:
        return None

    def rate_for(game_ids: set, exclude_missing: bool) -> float:
        rows = team_rows[team_rows["game_id"].isin(game_ids) & (team_rows["minutes"].fillna(0) > 0)]
        if exclude_missing:
            rows = rows[rows["player_name"].isin(rotation)]
        total_minutes = rows["minutes"].sum()
        if total_minutes <= 0:
            return 0.0
        stat_sum = rows["points"].sum() + rows["rebounds"].sum() + rows["assists"].sum()
        return float(stat_sum / total_minutes * 36)

    rate_full = rate_for(full_strength_games, exclude_missing=False)
    rate_missing = rate_for(missing_games, exclude_missing=False)
    if rate_full <= 0:
        return None

    pct_change = (rate_missing - rate_full) / rate_full * 100
    return {"pct_change": pct_change, "n_full": len(full_strength_games), "n_missing": len(missing_games)}


def main() -> int:
    df = load_played_minutes(Path("data/raw/player_boxscores_historical.csv"))
    teams = df["team"].unique()

    train_effects, test_effects = [], []
    for year in SEASONS:
        first_half, second_half = season_halves(df, year)
        if first_half.empty or second_half.empty:
            continue
        for team in teams:
            tr = pooled_effect(first_half, team)
            te = pooled_effect(second_half, team)
            if tr is None or te is None:
                continue
            train_effects.append(tr["pct_change"])
            test_effects.append(te["pct_change"])

    print(f"Real (team, season-half) pairs with a valid pooled effect in both halves: {len(train_effects)}")
    if len(train_effects) < 5:
        print("Too few real cases to validate.")
        return 0

    train_s = pd.Series(train_effects)
    test_s = pd.Series(test_effects)
    print(f"Real first-half pooled effect: mean={train_s.mean():+.2f}%, std={train_s.std():.2f}")
    print(f"Real second-half pooled effect: mean={test_s.mean():+.2f}%, std={test_s.std():.2f}")
    corr = train_s.corr(test_s)
    same_sign = ((train_s > 0) == (test_s > 0)).mean()
    print(f"Real out-of-sample correlation: {corr:.3f}")
    print(f"Real sign agreement: {same_sign:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
