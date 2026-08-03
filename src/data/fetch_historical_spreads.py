"""Bulk-fetch real historical spreads for garbage-time/blowout-risk modeling.

Matches our real historical games (2022-05-21 onward -- the confirmed real
start of the-odds-api's historical archive) to real historical spread
snapshots, so the garbage-time model can be fit against the literal real
market spread instead of only a team-rating proxy.

Resumable: game_ids already in the output CSV are skipped. Cost: confirmed
directly, 10 quota units per game (single market = spreads only), ~0 for the
events-listing call. ~1,268 real eligible games as of 2026-08-03 -> ~12,680
units, well within the 20,000/period paid-plan budget.

Usage:
    python -m src.data.fetch_historical_spreads
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.the-odds-api.com/v4/historical/sports/basketball_wnba"
OUTPUT_PATH = Path("data/raw/historical_spreads.csv")
BOXSCORE_PATH = Path("data/raw/player_boxscores_historical.csv")
ARCHIVE_START = "2022-05-21"

# Real WNBA team full names, as returned by the-odds-api, keyed by the
# abbreviation used in our own historical box score data. Only the teams
# active during the 2022-05-21+ archive window are needed here.
TEAM_FULL_NAME = {
    "ATL": "Atlanta Dream", "CHI": "Chicago Sky", "CON": "Connecticut Sun",
    "DAL": "Dallas Wings", "GS": "Golden State Valkyries", "IND": "Indiana Fever",
    "LV": "Las Vegas Aces", "LA": "Los Angeles Sparks", "MIN": "Minnesota Lynx",
    "NY": "New York Liberty", "PHX": "Phoenix Mercury", "POR": "Portland Fire",
    "SEA": "Seattle Storm", "TOR": "Toronto Tempo", "WSH": "Washington Mystics",
}


def _api_key() -> str:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY environment variable is not set.")
    return key


def load_real_games(boxscore_path: Path) -> pd.DataFrame:
    df = pd.read_csv(boxscore_path, usecols=["game_id", "team", "game_date"], dtype={"game_id": str})
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["game_date"] >= ARCHIVE_START]
    teams_per_game = df.groupby("game_id")["team"].unique()
    valid = teams_per_game[teams_per_game.apply(len) == 2]
    dates = df.groupby("game_id")["game_date"].first()

    rows = []
    for gid, teams in valid.items():
        a, b = teams
        if a not in TEAM_FULL_NAME or b not in TEAM_FULL_NAME:
            continue  # a team not active in the modern era (shouldn't occur post-2022, defensive)
        rows.append({"game_id": gid, "game_date": dates[gid], "team_a": a, "team_b": b})
    return pd.DataFrame(rows)


def load_existing_game_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {row["game_id"] for row in csv.DictReader(f)}


def fetch_events_for_date(date_str: str, session: requests.Session) -> list[dict]:
    resp = session.get(
        f"{BASE_URL}/events",
        params={"apiKey": _api_key(), "date": f"{date_str}T23:59:00Z"},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def fetch_spread_for_event(event_id: str, date_str: str, session: requests.Session) -> dict | None:
    resp = session.get(
        f"{BASE_URL}/events/{event_id}/odds",
        params={"apiKey": _api_key(), "regions": "us", "markets": "spreads", "date": f"{date_str}T23:59:00Z"},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json().get("data")
    if not data:
        return None
    home_team = data.get("home_team")
    spreads = []
    for bookmaker in data.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "spreads":
                continue
            for outcome in market.get("outcomes", []):
                if outcome.get("name") == home_team:
                    spreads.append(outcome.get("point"))
    if not spreads:
        return None
    return {"home_team": home_team, "home_spread_median": sorted(spreads)[len(spreads) // 2], "n_books": len(spreads)}


def fetch_all(output_path: Path, boxscore_path: Path) -> None:
    games = load_real_games(boxscore_path)
    existing_ids = load_existing_game_ids(output_path)
    remaining = games[~games["game_id"].isin(existing_ids)]
    print(f"{len(existing_ids)} games already fetched, {len(remaining)} remaining "
          f"(of {len(games)} real eligible games).", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["game_id", "game_date", "team_a", "team_b", "home_team", "home_spread_median", "n_books"]
    file_exists = output_path.exists()

    with requests.Session() as session, output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        by_date = remaining.groupby(remaining["game_date"].dt.date)
        done = 0
        for game_date, group in by_date:
            date_str = game_date.isoformat()
            try:
                events = fetch_events_for_date(date_str, session)
            except requests.RequestException as exc:
                print(f"  events fetch failed for {date_str}: {exc}", file=sys.stderr)
                continue

            for _, row in group.iterrows():
                full_a, full_b = TEAM_FULL_NAME[row["team_a"]], TEAM_FULL_NAME[row["team_b"]]
                match = next(
                    (e for e in events if {e.get("home_team"), e.get("away_team")} == {full_a, full_b}), None
                )
                if match is None:
                    continue
                try:
                    spread = fetch_spread_for_event(match["id"], date_str, session)
                except requests.RequestException as exc:
                    print(f"  odds fetch failed for {row['game_id']}: {exc}", file=sys.stderr)
                    continue
                if spread is None:
                    continue
                writer.writerow({
                    "game_id": row["game_id"], "game_date": date_str,
                    "team_a": row["team_a"], "team_b": row["team_b"],
                    **spread,
                })
                f.flush()
                done += 1
                time.sleep(0.2)

            if done and done % 100 == 0:
                print(f"  {done}/{len(remaining)} games done so far", file=sys.stderr)

    print(f"Done. {done} new games fetched.", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--boxscore-path", type=Path, default=BOXSCORE_PATH)
    args = parser.parse_args(argv)
    fetch_all(args.output, args.boxscore_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
