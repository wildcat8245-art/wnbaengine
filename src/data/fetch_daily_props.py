"""Fetch today's live WNBA player prop odds and save a dated snapshot.

Quota-aware: this key has a tight 500-request/month budget, and each
event's player props cost multiple units (one per market requested, ~3
here). This script checks the remaining quota after the (cheap) events
call and refuses to proceed past a safety floor rather than silently
draining the account.

Usage:
    python -m src.data.fetch_daily_props
    python -m src.data.fetch_daily_props --min-quota-floor 20
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

import requests

from src.data import odds_client
from src.data.wnba_client import iso_utc_to_us_game_date

OUTPUT_DIR = Path("data/raw/daily_props")


def fetch_today(min_quota_floor: int, output_dir: Path) -> int:
    with requests.Session() as session:
        events, quota = odds_client.get_events(session=session)
        print(f"{len(events)} upcoming/live WNBA events. Quota after events call: {quota}", file=sys.stderr)

        # get_events() returns ALL upcoming events, not just today's -- confirmed
        # directly (2026-08-01) it silently included games 1-2 days out, which
        # got saved into a file named after today's date and treated as today's
        # slate by predict_props.py. Filter to games whose real US game date
        # (UTC commence_time converted to US/Eastern, same conversion already
        # used for historical box scores) matches today before spending any
        # quota on their prop odds.
        today_us = date.today().isoformat()
        todays_events = [e for e in events if iso_utc_to_us_game_date(e.get("commence_time", "")) == today_us]
        skipped = len(events) - len(todays_events)
        if skipped:
            print(f"Skipping {skipped} event(s) not scheduled for today ({today_us}) US time.", file=sys.stderr)
        events = todays_events

        if quota.remaining is not None and quota.remaining < min_quota_floor:
            print(
                f"Remaining quota ({quota.remaining}) is below the safety floor "
                f"({min_quota_floor}) — stopping before fetching any prop odds.",
                file=sys.stderr,
            )
            return 1

        all_rows = []
        for event in events:
            if quota.remaining is not None and quota.remaining < min_quota_floor:
                print(
                    f"Quota floor reached mid-run ({quota.remaining} remaining) — "
                    f"stopping early, {len(all_rows)} rows collected so far.",
                    file=sys.stderr,
                )
                break
            try:
                event_odds, quota = odds_client.get_event_player_props(event["id"], session=session)
            except requests.RequestException as exc:
                print(f"  props fetch failed for event {event['id']}: {exc}", file=sys.stderr)
                continue
            rows = odds_client.flatten_props_to_rows(event_odds)
            all_rows.extend(rows)
            print(
                f"  {event.get('away_team')} @ {event.get('home_team')}: "
                f"{len(rows)} prop rows. Quota: {quota}",
                file=sys.stderr,
            )

    if not all_rows:
        print("No prop rows collected.", file=sys.stderr)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"props_{date.today().isoformat()}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-quota-floor", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    return fetch_today(args.min_quota_floor, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
