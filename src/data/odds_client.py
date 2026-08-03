"""Client for the-odds-api.com: live WNBA player prop odds.

Real, licensed player prop odds (not scraped) across multiple books
(FanDuel, DraftKings, BetRivers, BetOnline.ag confirmed present). Quota is
tight (500 requests/month on this key) — every call here reports the
remaining quota from response headers so callers can budget deliberately
instead of burning it on exploration.
"""

from __future__ import annotations

import os
from typing import Any

import requests

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_wnba"
PROP_MARKETS = "player_points,player_rebounds,player_assists,player_threes"
GAME_LINE_MARKETS = "spreads,totals"


class QuotaInfo:
    def __init__(self, headers: dict[str, str]) -> None:
        self.remaining = _int_or_none(headers.get("x-requests-remaining"))
        self.used = _int_or_none(headers.get("x-requests-used"))
        self.last_cost = _int_or_none(headers.get("x-requests-last"))

    def __repr__(self) -> str:
        return f"QuotaInfo(remaining={self.remaining}, used={self.used}, last_cost={self.last_cost})"


def _int_or_none(value: str | None) -> int | None:
    return int(value) if value is not None else None


def _api_key() -> str:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY environment variable is not set.")
    return key


def get_events(session: requests.Session | None = None) -> tuple[list[dict[str, Any]], QuotaInfo]:
    """List upcoming/live WNBA events. Cheap relative to per-event prop odds."""
    http = session or requests
    resp = http.get(
        f"{BASE_URL}/sports/{SPORT_KEY}/events/",
        params={"apiKey": _api_key()},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json(), QuotaInfo(resp.headers)


def get_event_player_props(
    event_id: str, session: requests.Session | None = None
) -> tuple[dict[str, Any], QuotaInfo]:
    """Fetch player prop odds (PTS/REB/AST, all books) for one event.

    Costs multiple quota units per call (one per market requested) — check
    QuotaInfo.remaining before calling this in a loop over many events.
    """
    http = session or requests
    resp = http.get(
        f"{BASE_URL}/sports/{SPORT_KEY}/events/{event_id}/odds/",
        params={"apiKey": _api_key(), "regions": "us", "markets": PROP_MARKETS},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json(), QuotaInfo(resp.headers)


def get_game_lines(session: requests.Session | None = None) -> tuple[list[dict[str, Any]], QuotaInfo]:
    """Fetch real game-level spread/total odds for all upcoming/live WNBA events in one call.

    Uses the bulk /sports/{sport}/odds/ endpoint rather than the per-event
    endpoint used for player props -- confirmed directly (2026-08-03) this
    costs 1 unit per market requested regardless of how many events come
    back (2 units total for spreads+totals across a full day's slate), not
    per-event like the player-prop endpoint.

    NOTE: this key's plan has no access to historical odds (confirmed
    directly: the-odds-api's /v4/historical/ endpoint returns
    HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN on this key). That means game
    total/spread can only ever be a live, display-time signal here -- there
    is no way to backfill real historical lines to train a model feature on
    or to backtest a total/spread-based adjustment against real past
    outcomes. Do not wire this into predict_props.py's actual probability/
    decision logic without first getting real historical access; it is
    informational context only (see predict_props.py's rationale text).
    """
    http = session or requests
    resp = http.get(
        f"{BASE_URL}/sports/{SPORT_KEY}/odds/",
        params={"apiKey": _api_key(), "regions": "us", "markets": GAME_LINE_MARKETS},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json(), QuotaInfo(resp.headers)


def flatten_game_lines_to_rows(events_odds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (event, bookmaker): the game total and the home team's spread.

    Collapses spreads' two-outcome (home/away) representation into a single
    signed home-team spread (negative = home favored), since that's the
    form useful for a per-player home/away adjustment later.
    """
    rows: list[dict[str, Any]] = []
    for event in events_odds:
        home_team = event.get("home_team")
        for bookmaker in event.get("bookmakers", []):
            total_point = None
            home_spread = None
            for market in bookmaker.get("markets", []):
                if market.get("key") == "totals":
                    outcomes = market.get("outcomes", [])
                    if outcomes:
                        total_point = outcomes[0].get("point")
                elif market.get("key") == "spreads":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == home_team:
                            home_spread = outcome.get("point")
            if total_point is None and home_spread is None:
                continue
            rows.append({
                "event_id": event.get("id"),
                "commence_time": event.get("commence_time"),
                "home_team": home_team,
                "away_team": event.get("away_team"),
                "bookmaker": bookmaker.get("key"),
                "game_total": total_point,
                "home_spread": home_spread,
            })
    return rows


def flatten_props_to_rows(event_odds: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (bookmaker, market, player, over/under)."""
    rows: list[dict[str, Any]] = []
    for bookmaker in event_odds.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            for outcome in market.get("outcomes", []):
                rows.append(
                    {
                        "event_id": event_odds.get("id"),
                        "commence_time": event_odds.get("commence_time"),
                        "home_team": event_odds.get("home_team"),
                        "away_team": event_odds.get("away_team"),
                        "bookmaker": bookmaker.get("key"),
                        "market": market.get("key"),
                        "player_name": outcome.get("description"),
                        "side": outcome.get("name"),
                        "line": outcome.get("point"),
                        "decimal_odds": outcome.get("price"),
                        "last_update": market.get("last_update"),
                    }
                )
    return rows
