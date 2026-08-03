"""Grade every saved predictions_*.csv board against real game outcomes.

Joins each saved prop pick to the player's real box score for that game
date (data/raw/player_boxscores_historical.csv) and checks whether the
side picked actually cleared the line. This is the closed loop that was
missing: predictions were being saved and odds/box scores were being
re-fetched, but nothing checked a specific past pick against what the
player actually did.

Only grades picks whose game date already has a real completed box score
on disk -- run fetch_historical_boxscores.py first to backfill recent
days if a date shows up ungraded.

Usage:
    python -m src.report.grade_predictions
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTIONS_GLOB = "data/processed/predictions_*.csv"
BOXSCORES_PATH = Path("data/raw/player_boxscores_historical.csv")
OUTPUT_PATH = Path("data/processed/graded_predictions.csv")

MARKET_TO_STAT_COL = {"points": "points", "rebounds": "rebounds", "assists": "assists", "tpm": "tpm"}


def _date_from_filename(path: Path) -> str | None:
    m = re.search(r"predictions_(\d{4}-\d{2}-\d{2})\.csv$", path.name)
    return m.group(1) if m else None


def load_real_outcomes(boxscores_path: Path) -> pd.DataFrame:
    df = pd.read_csv(boxscores_path, dtype={"game_id": str, "player_id": str})
    df["did_not_play"] = df["did_not_play"].astype(bool)
    played = df["did_not_play"].eq(False) & pd.to_numeric(df["minutes"], errors="coerce").fillna(0).gt(0)
    df = df[played].copy()
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["rebounds"] = pd.to_numeric(df["rebounds"], errors="coerce")
    df["assists"] = pd.to_numeric(df["assists"], errors="coerce")
    df["pra"] = df["points"] + df["rebounds"] + df["assists"]
    df["tpm"] = pd.to_numeric(
        df["threePointFieldGoalsMade-threePointFieldGoalsAttempted"].str.split("-", n=1).str[0], errors="coerce"
    )
    return df.set_index(["player_name", "game_date"])[["points", "rebounds", "assists", "pra", "tpm"]]


def grade_board(predictions_path: Path, board_date: str, outcomes: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(predictions_path)
    df["flag"] = df["flag"].fillna("")
    # One row per unique prop (dedupe across books -- the outcome doesn't depend on the book).
    df = df.drop_duplicates(subset=["player", "market", "market_line", "side"]).copy()

    df["board_date"] = board_date
    actual = []
    graded = []
    for _, row in df.iterrows():
        stat_col = MARKET_TO_STAT_COL.get(row["market"])
        key = (row["player"], board_date)
        if stat_col is None or key not in outcomes.index:
            actual.append(np.nan)
            graded.append(False)
            continue
        actual.append(outcomes.loc[key, stat_col])
        graded.append(True)

    df["actual_stat"] = actual
    df["graded"] = graded
    df["won"] = np.where(
        df["graded"],
        np.where(df["side"] == "Over", df["actual_stat"] > df["market_line"], df["actual_stat"] < df["market_line"]),
        np.nan,
    )
    df["recommended"] = df["kelly_stake"] > 0
    return df


def summarize(graded: pd.DataFrame) -> None:
    g = graded[graded["graded"]].copy()
    if g.empty:
        print("No graded rows yet -- none of the saved boards' game dates have a real box score on disk.")
        return

    print(f"\n{len(g)}/{len(graded)} saved picks have a real, completed game outcome on disk.")

    rec = g[g["recommended"]]
    if not rec.empty:
        win_rate = rec["won"].mean()
        avg_conf = rec["pick_pct"].str.extract(r"([\d.]+)%")[0].astype(float).mean() / 100
        print(f"\nRecommended bets: {len(rec)}, real win rate {win_rate:.1%}, "
              f"average stated confidence {avg_conf:.1%} (gap {avg_conf - win_rate:+.1%}).")

        # Real bankroll if every recommended bet had actually been placed at its own quoted odds.
        bankroll = 1000.0
        for _, row in rec.sort_values("board_date").iterrows():
            stake = row["kelly_stake"]
            if row["won"]:
                bankroll += stake * (row["decimal_odds"] - 1)
            else:
                bankroll -= stake
        print(f"Real bankroll if every recommended stake had been placed: ${bankroll:.2f} "
              f"(started at $1000, {len(rec)} bets).")

        print("\nBy market:")
        for market, mg in rec.groupby("market"):
            print(f"  {market}: {len(mg)} bets, win rate {mg['won'].mean():.1%}")

    flagged = g[g["flag"] != ""]
    if not flagged.empty:
        would_have = flagged["won"].mean()
        print(f"\nFlagged/stay-away picks with a real outcome: {len(flagged)}, "
              f"would-have-won rate {would_have:.1%} (these were excluded regardless of EV -- "
              f"this checks whether the guard was actually right to exclude them).")


def main() -> int:
    outcomes = load_real_outcomes(BOXSCORES_PATH)

    all_graded = []
    for path_str in sorted(glob.glob(PREDICTIONS_GLOB)):
        path = Path(path_str)
        board_date = _date_from_filename(path)
        if board_date is None:
            continue
        all_graded.append(grade_board(path, board_date, outcomes))

    if not all_graded:
        print("No predictions_*.csv files found.")
        return 1

    result = pd.concat(all_graded, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(result)} graded rows (across {result['board_date'].nunique()} board(s)) to {OUTPUT_PATH}")

    summarize(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
