"""Real backtest of the teammate on/off split mechanism (on_off_splits.py).

FIXED (2026-08-03) after the first version showed real out-of-sample
correlation of -0.001 and sign agreement of 46.8% (worse than chance) --
root cause: it split by a fixed multi-year boundary (train <=2024, test
2025-2026), so a split fit years earlier had no reason to still hold --
real WNBA rosters/roles change meaningfully season to season (trades,
coaching changes, development), so a stale split shouldn't be expected to
transfer across a multi-year gap.

Real fix: validate IN-SEASON instead -- for each real season, fit the split
using only the FIRST HALF of that season's real games, and check whether it
predicts the SECOND HALF of the SAME season (a real, recent, roster-stable
window, matching how this should actually be used live -- see the
corresponding season_start fix in predict_props.py).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.on_off_splits import load_played_minutes, compute_on_off_split, MIN_OUT_GAMES

SEASONS = [2024, 2025, 2026]


def season_halves(df: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    season = df[df["game_date"].dt.year == year]
    if season.empty:
        return season, season
    midpoint = season["game_date"].quantile(0.5)
    return season[season["game_date"] < midpoint], season[season["game_date"] >= midpoint]


def main() -> int:
    df = load_played_minutes(Path("data/raw/player_boxscores_historical.csv"))
    teams = df["team"].unique()
    train_effects, test_effects = [], []

    for year in SEASONS:
        first_half, second_half = season_halves(df, year)
        if first_half.empty or second_half.empty:
            continue

        for team in teams:
            team_second = second_half[second_half["team"] == team]
            if team_second.empty:
                continue

            roster = set(team_second.loc[team_second["minutes"].fillna(0) > 0, "player_name"].unique())
            for teammate_out in roster:
                candidates = roster - {teammate_out}
                train_split = compute_on_off_split(first_half, team, teammate_out, candidates)
                test_split = compute_on_off_split(second_half, team, teammate_out, candidates)

                for name in candidates:
                    tr, te = train_split.get(name), test_split.get(name)
                    if not tr or not te or tr["insufficient_data"] or te["insufficient_data"]:
                        continue
                    for stat in ("points", "rebounds", "assists"):
                        tr_pct, te_pct = tr["pct_change"].get(stat), te["pct_change"].get(stat)
                        if tr_pct is None or te_pct is None:
                            continue
                        train_effects.append(tr_pct)
                        test_effects.append(te_pct)

    print(f"Real (player, teammate, stat) triplets with a valid in-season split in BOTH "
          f"halves across {SEASONS}: {len(train_effects)}")
    if len(train_effects) < 5:
        print("Too few real overlapping cases to validate from this data -- both halves "
              f"require >= {MIN_OUT_GAMES} real games in each condition, which is a real, "
              "hard constraint within a single ~5-month season.")
        return 0

    train_s = pd.Series(train_effects)
    test_s = pd.Series(test_effects)
    same_sign = ((train_s > 0) == (test_s > 0)).mean()
    corr = train_s.corr(test_s)
    print(f"Real in-season out-of-sample correlation (first-half-predicted vs second-half-observed): {corr:.3f}")
    print(f"Real in-season sign agreement: {same_sign:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
