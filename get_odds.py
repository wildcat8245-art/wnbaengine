#!/usr/bin/env python3
"""Fetch live odds from DraftKings and return them as a pandas DataFrame.

DraftKings exposes an unofficial JSON sportsbook API. This script queries the
event-group endpoint for a given sport/league, flattens the nested markets and
outcomes into tabular rows, and prints (or saves) the resulting odds table.

Usage:
    python get_odds.py                       # WNBA moneyline/spread/total
    python get_odds.py --event-group 42648   # a specific event group id
    python get_odds.py --output odds.csv     # write results to CSV

Notes:
    The DraftKings API is undocumented and may change without notice. It is used
    here for personal/educational purposes only; respect DraftKings' terms of
    service and any applicable rate limits.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import pandas as pd
import requests

# DraftKings sportsbook API base. Event groups map to a league/competition.
# 94682 is the WNBA event group at time of writing.
DK_API_BASE = "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups"
DEFAULT_EVENT_GROUP = "94682"  # WNBA

# A browser-like User-Agent avoids trivial bot filtering.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://sportsbook.draftkings.com/",
}

REQUEST_TIMEOUT = 15  # seconds


def fetch_event_group(event_group: str, session: requests.Session | None = None) -> dict[str, Any]:
    """Fetch the raw event-group payload from DraftKings.

    Args:
        event_group: The DraftKings event group id (league identifier).
        session: Optional requests.Session for connection reuse.

    Returns:
        The parsed JSON response as a dict.

    Raises:
        requests.HTTPError: If the request returns a non-2xx status.
    """
    url = f"{DK_API_BASE}/{event_group}?format=json"
    http = session or requests
    response = http.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def parse_odds(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten a DraftKings event-group payload into a tidy odds DataFrame.

    The payload nests data as:
        eventGroup -> offerCategories -> offerSubcategoryDescriptors
            -> offerSubcategory -> offers -> outcomes

    Each output row is a single betting outcome (e.g. one team's moneyline)
    with its associated event, market, line, and odds.

    Args:
        payload: The JSON dict returned by :func:`fetch_event_group`.

    Returns:
        A DataFrame with one row per outcome. Empty if no offers are present.
    """
    event_group = payload.get("eventGroup", {})

    # Build a lookup of event metadata (teams, start time) keyed by event id.
    events: dict[Any, dict[str, Any]] = {}
    for event in event_group.get("events", []):
        events[event.get("eventId")] = {
            "event_name": event.get("name"),
            "start_date": event.get("startDate"),
            "team1": event.get("teamName1"),
            "team2": event.get("teamName2"),
        }

    rows: list[dict[str, Any]] = []
    for category in event_group.get("offerCategories", []):
        category_name = category.get("name")
        for descriptor in category.get("offerSubcategoryDescriptors", []):
            subcategory = descriptor.get("offerSubcategory")
            if not subcategory:
                continue
            market_name = subcategory.get("name")
            # `offers` is a list of lists (one inner list per event).
            for offer_group in subcategory.get("offers", []):
                for offer in offer_group:
                    event_id = offer.get("eventId")
                    event_meta = events.get(event_id, {})
                    for outcome in offer.get("outcomes", []):
                        rows.append(
                            {
                                "event_id": event_id,
                                "event_name": event_meta.get("event_name"),
                                "start_date": event_meta.get("start_date"),
                                "category": category_name,
                                "market": market_name,
                                "label": offer.get("label"),
                                "outcome": outcome.get("label"),
                                "line": outcome.get("line"),
                                "odds_american": outcome.get("oddsAmerican"),
                                "odds_decimal": outcome.get("oddsDecimal"),
                            }
                        )

    return pd.DataFrame(rows)


def get_odds(event_group: str = DEFAULT_EVENT_GROUP) -> pd.DataFrame:
    """Fetch and parse DraftKings odds for a given event group.

    Args:
        event_group: The DraftKings event group id. Defaults to WNBA.

    Returns:
        A DataFrame of odds outcomes.
    """
    with requests.Session() as session:
        payload = fetch_event_group(event_group, session=session)
    return parse_odds(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch live odds from DraftKings as a pandas DataFrame."
    )
    parser.add_argument(
        "--event-group",
        default=DEFAULT_EVENT_GROUP,
        help=f"DraftKings event group id (default: {DEFAULT_EVENT_GROUP}, WNBA).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the odds table as CSV.",
    )
    args = parser.parse_args(argv)

    try:
        df = get_odds(args.event_group)
    except requests.RequestException as exc:
        print(f"Error fetching odds from DraftKings: {exc}", file=sys.stderr)
        return 1

    if df.empty:
        print("No odds found for the requested event group.", file=sys.stderr)
        return 0

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} rows to {args.output}")
    else:
        # Show the full table without pandas truncating columns.
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
