"""Pull real substitution events for a recent real-game sample, for the
Garbage-Time & Blowout Risk layer's substitution-timing analysis.

Confirmed directly (2026-08-03): substitution plays carry real structured
participants (participants[0] = player entering, participants[1] = player
exiting -- verified against our own real player_id/player_name data, not
assumed), plus the real score/period/clock at that exact moment. Combined
with each game's real box-score `starter` flag, this gives a genuine,
data-driven answer to "at what real margin do coaches actually pull
starters" -- not a guessed threshold.

Deliberately scoped to a recent real sample (2025-2026 seasons, ~535 games)
rather than the full 2015-2026 history -- this is a validating/enriching
analysis of CURRENT coaching behavior, not a per-player training feature
needing full career coverage, and re-pulling all ~2,768 games a second time
(the first pull already discarded these events) would cost real time/quota
disproportionate to what this specific question needs.

Usage:
    python -m src.data.fetch_substitution_events
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import requests

from src.data import wnba_client

OUTPUT_PATH = Path("data/raw/substitution_events_recent.csv")
BOXSCORE_PATH = Path("data/raw/player_boxscores_historical.csv")
DEFAULT_SEASON_START = "2025-01-01"


def parse_substitution_events(summary: dict, game_id: str, starters: set[str]) -> list[dict]:
    rows = []
    for play in summary.get("plays", []):
        if play.get("type", {}).get("text") != "Substitution":
            continue
        participants = play.get("participants", [])
        if len(participants) < 2:
            continue
        player_out_id = participants[1].get("athlete", {}).get("id")
        if not player_out_id:
            continue
        rows.append({
            "game_id": game_id,
            "player_out_id": player_out_id,
            "player_out_is_starter": player_out_id in starters,
            "period": play.get("period", {}).get("number"),
            "clock": play.get("clock", {}).get("displayValue"),
            "home_score": play.get("homeScore"),
            "away_score": play.get("awayScore"),
            "margin_abs": abs((play.get("homeScore") or 0) - (play.get("awayScore") or 0)),
        })
    return rows


def load_existing_game_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {row["game_id"] for row in csv.DictReader(f)}


def fetch_all(output_path: Path, boxscore_path: Path, season_start: str) -> None:
    box = pd.read_csv(boxscore_path, dtype={"game_id": str, "player_id": str})
    box["game_date"] = pd.to_datetime(box["game_date"])
    recent = box[box["game_date"] >= season_start]
    game_ids = sorted(recent["game_id"].unique())

    starters_by_game = (
        recent[recent["starter"] == True]  # noqa: E712
        .groupby("game_id")["player_id"].apply(set)
        .to_dict()
    )

    existing_ids = load_existing_game_ids(output_path)
    remaining = [g for g in game_ids if g not in existing_ids]
    print(f"{len(existing_ids)}/{len(game_ids)} games already fetched, {len(remaining)} remaining.",
          file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "game_id", "player_out_id", "player_out_is_starter", "period", "clock",
        "home_score", "away_score", "margin_abs",
    ]
    file_exists = output_path.exists()
    limiter = wnba_client.RateLimiter()

    with requests.Session() as session, output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i, game_id in enumerate(remaining):
            limiter.wait()
            try:
                resp = session.get(
                    f"{wnba_client.BASE_URL}/wnbasummary",
                    params={"gameId": game_id},
                    headers=wnba_client._headers(),
                    timeout=20,
                )
                resp.raise_for_status()
                summary = resp.json()
            except requests.RequestException as exc:
                print(f"  summary fetch failed for {game_id}: {exc}", file=sys.stderr)
                continue

            rows = parse_substitution_events(summary, game_id, starters_by_game.get(game_id, set()))
            for row in rows:
                writer.writerow(row)
            f.flush()

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(remaining)} games done", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--boxscore-path", type=Path, default=BOXSCORE_PATH)
    parser.add_argument("--season-start", default=DEFAULT_SEASON_START)
    args = parser.parse_args(argv)
    fetch_all(args.output, args.boxscore_path, args.season_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
