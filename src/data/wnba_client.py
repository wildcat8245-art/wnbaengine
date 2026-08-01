"""Client for wnba-api on RapidAPI: schedules and per-player box scores.

Real per-game player box scores are available back to at least the 2015
season (verified directly against the live API, not assumed). The 2020
season has a real gap in June because that season started late (COVID),
not a data hole.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests

API_HOST = "wnba-api.p.rapidapi.com"
BASE_URL = f"https://{API_HOST}"

_UTC = ZoneInfo("UTC")
_US_EASTERN = ZoneInfo("America/New_York")


def iso_utc_to_us_game_date(iso_timestamp: str) -> str:
    """Convert an ESPN-style UTC timestamp to the real US game date.

    Confirmed by direct testing: games are stamped in UTC, and a US-evening
    game can be stamped at/after UTC midnight, landing on the next calendar
    day if truncated naively (e.g. "2015-06-06T00:00Z" is actually the
    evening of June 5 in US Eastern). Converting to US Eastern first before
    taking the date is what actually recovers the correct game night.
    """
    if not iso_timestamp:
        return ""
    dt = datetime.strptime(iso_timestamp, "%Y-%m-%dT%H:%MZ").replace(tzinfo=_UTC)
    return dt.astimezone(_US_EASTERN).date().isoformat()


def _headers() -> dict[str, str]:
    key = os.environ.get("RAPIDAPI_KEY")
    if not key:
        raise RuntimeError("RAPIDAPI_KEY environment variable is not set.")
    return {"x-rapidapi-key": key, "x-rapidapi-host": API_HOST}


def get_schedule(year: int, month: int, day: int, session: requests.Session | None = None) -> dict[str, Any]:
    """Fetch the schedule around a given day. Returns {yyyymmdd: [games]}.

    Confirmed by direct testing: omitting `day` does NOT return the whole
    month — it silently returns some fixed ~9-day window instead, which
    would silently undercount/misattribute games. A day must always be
    passed; callers iterate every day of a month to get full coverage.
    Also confirmed: the response can include a neighboring day's games
    (the query day plus the day before), so callers must dedupe by game id
    across days rather than trust the outer dict key as ground truth.
    """
    http = session or requests
    resp = http.get(
        f"{BASE_URL}/wnbaschedule",
        params={"year": year, "month": f"{month:02d}", "day": f"{day:02d}"},
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def get_boxscore(game_id: str, session: requests.Session | None = None) -> dict[str, Any]:
    """Fetch the full box score (teams + per-player stats) for one game."""
    http = session or requests
    resp = http.get(
        f"{BASE_URL}/wnbabox",
        params={"gameId": game_id},
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def parse_boxscore_to_player_rows(boxscore: dict[str, Any], game_date: str) -> list[dict[str, Any]]:
    """Flatten a wnbabox payload into one row per player-game.

    Stat keys/values are aligned positionally via each team block's own
    `keys` list rather than assumed to be in a fixed order, since the API
    does not document a stable column order guarantee.
    """
    rows: list[dict[str, Any]] = []
    game_id = boxscore.get("id")

    home_away_by_team = {
        block.get("team", {}).get("abbreviation"): block.get("homeAway")
        for block in boxscore.get("teams", [])
    }

    for team_block in boxscore.get("players", []):
        team = team_block.get("team", {})
        team_abbr = team.get("abbreviation")
        for stat_group in team_block.get("statistics", []):
            keys = stat_group.get("keys", [])
            for athlete_entry in stat_group.get("athletes", []):
                athlete = athlete_entry.get("athlete", {})
                stats = athlete_entry.get("stats", [])
                row = {
                    "game_id": game_id,
                    "game_date": game_date,
                    "team": team_abbr,
                    "home_away": home_away_by_team.get(team_abbr),
                    "player_id": athlete.get("id"),
                    "player_name": athlete.get("displayName"),
                    "position": athlete.get("position", {}).get("abbreviation"),
                    "starter": athlete_entry.get("starter", False),
                    "did_not_play": athlete_entry.get("didNotPlay", False),
                }
                for key, value in zip(keys, stats):
                    row[key] = value
                rows.append(row)

    return rows


EXCLUDED_SEASON_SLUGS = {"preseason"}


def iter_completed_game_ids(
    year: int, month: int, limiter: "RateLimiter | None" = None, session: requests.Session | None = None
) -> list[tuple[str, str]]:
    """Return [(game_id, iso_date), ...] for all completed regular/post-season games in a month.

    Iterates every day of the month explicitly (month-only queries do not
    return full-month data, see get_schedule's docstring) and dedupes by
    game id, since adjacent days' queries overlap. `iso_date` comes from
    each game's own `date` field, not the response's outer dict key.

    Preseason games are excluded (confirmed via each game's own
    `season.slug` field: "preseason" vs "regular-season"/"post-season") —
    preseason rotations/rest patterns aren't representative of real games
    and have no real betting markets attached.
    """
    import calendar

    seen: dict[str, str] = {}
    _, days_in_month = calendar.monthrange(year, month)

    failed_days: list[int] = []
    for day in range(1, days_in_month + 1):
        if limiter is not None:
            limiter.wait()
        try:
            schedule = get_schedule(year, month, day, session=session)
        except requests.RequestException as exc:
            # A single bad day must not sacrifice the whole month. Because
            # each day's response also includes the prior day (see
            # get_schedule's docstring), the neighboring day's successful
            # call will often still cover most of what a failed day missed.
            print(f"    schedule fetch failed for {year}-{month:02d}-{day:02d}: {exc}", file=sys.stderr)
            failed_days.append(day)
            continue
        for games in schedule.values():
            for game in games:
                slug = game.get("season", {}).get("slug")
                if (
                    game.get("completed")
                    and slug not in EXCLUDED_SEASON_SLUGS
                    and game.get("id") not in seen
                ):
                    seen[game["id"]] = game.get("date", "")

    if failed_days:
        print(
            f"  {year}-{month:02d}: {len(failed_days)} day(s) failed outright: {failed_days} "
            "(neighboring days may have still covered some of their games)",
            file=sys.stderr,
        )

    return list(seen.items())


class RateLimiter:
    """Tiny courtesy delay between requests; wnba-api has no documented per-second limit."""

    def __init__(self, delay_seconds: float = 0.3) -> None:
        self.delay_seconds = delay_seconds
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        self._last = time.monotonic()
