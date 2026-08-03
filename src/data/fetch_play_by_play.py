"""Bulk-fetch real shot-location events from play-by-play for every historical game.

Needed for the shot-zone breakdown of the Shooting Efficiency layer (rim /
short mid-range / long mid-range / 3PT). Confirmed directly (2026-08-03):
wnba-api's /wnbasummary endpoint returns a real "plays" array per game with
shootingPlay/scoringPlay/pointsAttempted flags and a real shot distance in
the play's own "text" field (e.g. "Diana Taurasi makes 4-foot two point
shot") -- confirmed the named shooter matches participants[0]'s athlete id
against our own real box score player_id/player_name data.

Free throws also carry shootingPlay=True (confirmed directly) -- excluded
via pointsAttempted in (2, 3), which free throws (pointsAttempted=1) fail.

Resumable: games already present in the output CSV are skipped, matching
fetch_historical_boxscores.py's pattern exactly.

Usage:
    python -m src.data.fetch_play_by_play
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import requests

from src.data import wnba_client

OUTPUT_PATH = Path("data/raw/shot_events_historical.csv")
BOXSCORE_PATH = Path("data/raw/player_boxscores_historical.csv")

_DISTANCE_RE = re.compile(r"(\d+)-foot")


_RIM_TYPE_KEYWORDS = ("layup", "dunk", "tip")


def classify_zone(text: str, type_text: str, distance_feet: float | None) -> str:
    """Real distinction confirmed directly (2026-08-03): pointsAttempted is
    NOT usable to tell a missed 2 from a missed 3 -- it reads 0 for every
    miss regardless of shot value (only meaningful on makes, where it
    duplicates scoreValue). The play's own free-text description, however,
    reliably labels three-point attempts explicitly on BOTH makes and
    misses (e.g. "Jasmine Thomas misses 26-foot three point jumper") --
    used as the primary signal here instead of distance/pointsAttempted.

    Second real bug found and fixed the same way (caught by an honest
    sanity check, not assumed correct): layups/dunks routinely have NO
    distance in the text at all ("misses layup", "makes dunk") -- a
    distance-only rule silently dumped nearly all real rim attempts into
    "unknown" instead of "rim", leaving "rim" as a tiny, biased sample of
    unusual shots that happened to name a short distance, and its measured
    make rate came out LOWER than mid-range -- backwards from real
    basketball. Fixed by checking the play's own TYPE text for
    layup/dunk/tip keywords first (a reliable, explicit signal that doesn't
    depend on distance being mentioned at all).
    """
    text_lower = text.lower()
    type_lower = type_text.lower()
    if "three point" in text_lower or "3pt" in text_lower:
        return "three"
    if any(kw in type_lower for kw in _RIM_TYPE_KEYWORDS):
        return "rim"
    if distance_feet is None:
        return "unknown"
    if distance_feet > 22:
        return "three"
    if distance_feet <= 3:
        return "rim"
    if distance_feet <= 15:
        return "short_midrange"
    return "long_midrange"


def parse_shot_events(summary: dict, game_id: str) -> list[dict]:
    rows = []
    for play in summary.get("plays", []):
        if not play.get("shootingPlay"):
            continue
        text = play.get("text", "")
        type_text = play.get("type", {}).get("text", "")
        if "free throw" in type_text.lower() or "free throw" in text.lower():
            continue  # free throws also carry shootingPlay=True; not a field goal attempt

        participants = play.get("participants", [])
        if not participants:
            continue
        player_id = participants[0].get("athlete", {}).get("id")
        if not player_id:
            continue

        m = _DISTANCE_RE.search(text)
        distance_feet = float(m.group(1)) if m else None

        rows.append({
            "game_id": game_id,
            "player_id": player_id,
            "made": bool(play.get("scoringPlay")),
            "distance_feet": distance_feet,
            "zone": classify_zone(text, type_text, distance_feet),
            "period": play.get("period", {}).get("number"),
        })
    return rows


def load_existing_game_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["game_id"] for row in reader}


def fetch_all(output_path: Path, boxscore_path: Path) -> None:
    all_game_ids = set(
        pd_game_id for pd_game_id in
        __import__("pandas").read_csv(boxscore_path, usecols=["game_id"], dtype={"game_id": str})["game_id"].unique()
    )
    existing_ids = load_existing_game_ids(output_path)
    remaining = sorted(all_game_ids - existing_ids)
    print(f"{len(existing_ids)}/{len(all_game_ids)} games already fetched, {len(remaining)} remaining.",
          file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["game_id", "player_id", "made", "distance_feet", "zone", "period"]
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

            rows = parse_shot_events(summary, game_id)
            for row in rows:
                writer.writerow(row)
            f.flush()

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(remaining)} games done ({len(existing_ids) + i + 1} total)", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--boxscore-path", type=Path, default=BOXSCORE_PATH)
    args = parser.parse_args(argv)

    fetch_all(args.output, args.boxscore_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
