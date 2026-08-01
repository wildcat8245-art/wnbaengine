"""Bulk-fetch real historical WNBA player box scores across multiple seasons.

Builds the dataset needed for genuine chronological train/validate/test
splits (paper-style: e.g. train on old seasons, validate on one, test on
the most recent). Resumable: games already present in the output CSV are
skipped, so a partial/interrupted run can just be re-launched.

Usage:
    python -m src.data.fetch_historical_boxscores --start-year 2015 --end-year 2026
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import requests

from src.data import wnba_client

OUTPUT_PATH = Path("data/raw/player_boxscores_historical.csv")

# WNBA regular season + playoffs run roughly May through mid-October.
SEASON_MONTHS = range(5, 11)


def load_existing_game_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["game_id"] for row in reader}


def fetch_all(start_year: int, end_year: int, output_path: Path) -> None:
    existing_ids = load_existing_game_ids(output_path)
    print(f"{len(existing_ids)} games already in {output_path}, will skip those.", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    limiter = wnba_client.RateLimiter()

    fieldnames: list[str] | None = None
    if file_exists:
        with output_path.open(newline="", encoding="utf-8") as f:
            fieldnames = next(csv.reader(f), None)

    with requests.Session() as session, output_path.open("a", newline="", encoding="utf-8") as f:
        writer = None
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

        for year in range(start_year, end_year + 1):
            for month in SEASON_MONTHS:
                try:
                    game_ids = wnba_client.iter_completed_game_ids(year, month, limiter=limiter, session=session)
                except requests.RequestException as exc:
                    print(f"Schedule fetch failed for {year}-{month:02d}: {exc}", file=sys.stderr)
                    continue

                if not game_ids:
                    continue
                print(f"{year}-{month:02d}: {len(game_ids)} completed games", file=sys.stderr)

                for game_id, iso_date in game_ids:
                    if game_id in existing_ids:
                        continue
                    limiter.wait()
                    try:
                        box = wnba_client.get_boxscore(game_id, session=session)
                    except requests.RequestException as exc:
                        print(f"  box score failed for {game_id}: {exc}", file=sys.stderr)
                        continue

                    game_date = wnba_client.iso_utc_to_us_game_date(iso_date)
                    rows = wnba_client.parse_boxscore_to_player_rows(box, game_date)
                    if not rows:
                        continue

                    if writer is None:
                        fieldnames = list(rows[0].keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()

                    for row in rows:
                        unknown = set(row) - set(fieldnames)
                        if unknown:
                            raise ValueError(
                                f"game {game_id} introduced new box score columns not in the "
                                f"CSV header: {unknown}. Rerun with a fresh output file so the "
                                f"header can include them, rather than writing a mismatched row."
                            )
                        writer.writerow(row)

                    existing_ids.add(game_id)
                    f.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    fetch_all(args.start_year, args.end_year, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
