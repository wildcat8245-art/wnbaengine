"""Render the latest predict_props.py output into the standalone HTML board.

Groups the flat predictions CSV (one row per player/market/line/side/book)
into one row per unique prop, keeping the best-odds book as the headline
figure and every book's odds available underneath. Fills templates/board_
template.html's placeholders and writes a self-contained HTML file --
no external requests, safe to publish as a Claude Artifact.

Usage:
    python -m src.report.build_board
    python -m src.report.build_board --date 2026-08-03
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import date
from pathlib import Path

import pandas as pd

TEMPLATE_PATH = Path("templates/board_template.html")
BANKROLL = 1000.0
KELLY_FRACTION = 0.3
EV_THRESHOLD = 1.05


def _latest_file(pattern: str, explicit_date: str | None) -> Path:
    if explicit_date:
        path = Path(pattern.replace("*", explicit_date))
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        return path
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern}")
    return Path(matches[-1])


def _player_game_map(props_path: Path) -> dict[str, str]:
    props = pd.read_csv(props_path)
    props = props.drop_duplicates("player_name")
    return {
        row["player_name"]: f"{row['away_team']} @ {row['home_team']}"
        for _, row in props.iterrows()
    }


def build_records(predictions_path: Path, props_path: Path) -> list[dict]:
    df = pd.read_csv(predictions_path)
    for col in ["flag", "injury_note", "usage_vacuum"]:
        df[col] = df[col].fillna("")
    df["injury_status"] = df["injury_status"].fillna("no report")

    player_game = _player_game_map(props_path)

    records = []
    key_cols = ["player", "market", "market_line", "side"]
    for (player, market, line, side), g in df.groupby(key_cols, sort=False):
        g = g.sort_values("ev", ascending=False)
        best = g.iloc[0]
        books = sorted(
            (
                {
                    "book": r["book"],
                    "odds": float(r["decimal_odds"]),
                    "ev": round(float(r["ev"]), 3),
                    "stake": float(r["kelly_stake"]),
                }
                for _, r in g.iterrows()
            ),
            key=lambda b: -b["ev"],
        )
        records.append({
            "player": player,
            "game": player_game.get(player, ""),
            "market": market,
            "line": float(line),
            "side": side,
            "best_book": best["book"],
            "best_odds": float(best["decimal_odds"]),
            "books": books,
            "projection": float(best["system_projection"]),
            "projection_type": best["projection_type"],
            "mean20": None if pd.isna(best["real_last20_mean"]) else float(best["real_last20_mean"]),
            "diff_vs_line": float(best["diff_vs_line"]),
            "confidence_pct": best["pick_pct"],
            "model_prob_over_pct": best["model_prob_over_pct"],
            "mc_over_pct": best["monte_carlo_over_pct"],
            "mc_pool": best["monte_carlo_pool"],
            "ev": round(float(best["ev"]), 3),
            "stake": float(best["kelly_stake"]),
            "flag": best["flag"],
            "injury_status": best["injury_status"],
            "injury_note": best["injury_note"],
            "usage_vacuum": best["usage_vacuum"],
            "rationale": best["rationale"],
            "last5": float(best["stat_per100_last5"]),
            "last10": float(best["stat_per100_last10"]),
            "minutes_trend": best["minutes_last5_vs_last10"],
            "recommended": bool((g["kelly_stake"] > 0).any()),
        })

    records.sort(key=lambda r: -r["ev"])
    return records


def render(board_date: str, records: list[dict]) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    meta = f"${BANKROLL:,.0f} bankroll · {KELLY_FRACTION}x Kelly · EV>{EV_THRESHOLD}"
    html = template.replace("__BOARD_DATE__", board_date)
    html = html.replace("__BOARD_META__", meta)
    html = html.replace("__BOARD_DATA__", json.dumps(records))
    return html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="YYYY-MM-DD; defaults to the latest snapshot on disk")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output HTML path; defaults to data/processed/board_<date>.html",
    )
    args = parser.parse_args(argv)

    predictions_path = _latest_file("data/processed/predictions_*.csv", args.date)
    board_date = args.date or predictions_path.stem.replace("predictions_", "")
    props_path = _latest_file("data/raw/daily_props/props_*.csv", board_date)

    records = build_records(predictions_path, props_path)
    html = render(board_date, records)

    output = args.output or Path(f"data/processed/board_{board_date}.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Wrote {len(records)} unique props to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
