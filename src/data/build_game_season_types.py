"""Walk the schedule (cheap: no box scores) to map every game_id to its season type.

Used to retroactively filter preseason games out of a box score dataset
that was collected before EXCLUDED_SEASON_SLUGS existed, without having to
re-fetch any box scores (the expensive part) — only schedule calls.

Usage:
    python -m src.data.build_game_season_types --start-year 2015 --end-year 2026
"""

from __future__ import annotations

import argparse
import calendar
import csv
import sys
from pathlib import Path

import requests

from src.data import wnba_client
from src.data.fetch_historical_boxscores import SEASON_MONTHS

OUTPUT_PATH = Path("data/raw/game_season_types.csv")


def build_mapping(start_year: int, end_year: int) -> list[dict[str, str]]:
    limiter = wnba_client.RateLimiter()
    rows: dict[str, dict[str, str]] = {}

    with requests.Session() as session:
        for year in range(start_year, end_year + 1):
            for month in SEASON_MONTHS:
                _, days_in_month = calendar.monthrange(year, month)
                for day in range(1, days_in_month + 1):
                    limiter.wait()
                    try:
                        schedule = wnba_client.get_schedule(year, month, day, session=session)
                    except requests.RequestException as exc:
                        print(f"    failed {year}-{month:02d}-{day:02d}: {exc}", file=sys.stderr)
                        continue
                    for games in schedule.values():
                        for game in games:
                            gid = game.get("id")
                            if gid and gid not in rows:
                                rows[gid] = {
                                    "game_id": gid,
                                    "season_slug": game.get("season", {}).get("slug", ""),
                                    "completed": game.get("completed", False),
                                    "game_date_us": wnba_client.iso_utc_to_us_game_date(game.get("date", "")),
                                }
                print(f"{year}-{month:02d}: {len(rows)} games mapped so far", file=sys.stderr)

    return list(rows.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    rows = build_mapping(args.start_year, args.end_year)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["game_id", "season_slug", "completed", "game_date_us"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} game->season_type rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
