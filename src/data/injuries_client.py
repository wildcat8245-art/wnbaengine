"""Client for wnba-api's real injury report endpoint.

Real, live injury data (ESPN-sourced), not scraped -- confirmed directly by
querying the API (2026-08-02T14:13Z snapshot returned 14 teams' worth of
real entries with dated player-level notes). Uses the same RAPIDAPI_KEY as
wnba_client.py. Never wired into this codebase before 2026-08-02; the
prior WNBA project (deleted 2026-07-31) had an equivalent
`get_injury_reports()` against what its notes call "the richer RapidAPI
endpoint" -- this is that same endpoint, rebuilt here from scratch.
"""

from __future__ import annotations

from typing import Any

import requests

from src.data.wnba_client import API_HOST, BASE_URL, _headers

# The structured `status` field only carries two coarse values in practice
# (confirmed directly): "Out" and "Day-To-Day". Finer-grained language
# (questionable/probable/doubtful) only shows up in the free-text
# shortComment/longComment, not as its own structured field. OUT_STATUSES
# is what actually blocks a bet; everything else is a soft flag only.
OUT_STATUSES = {"out"}
OUT_FANTASY_ABBR = {"OUT", "OFS"}  # OFS = out for season


def fetch_injuries(session: requests.Session | None = None) -> dict[str, Any]:
    http = session or requests
    resp = http.get(f"{BASE_URL}/injuries", headers=_headers(), timeout=20)
    resp.raise_for_status()
    return resp.json()


def build_injury_map(data: dict[str, Any], teams: set[str] | None = None) -> dict[str, dict[str, Any]]:
    """player_display_name -> {team, status, is_out, type, short_comment, date}.

    `teams`, if given, restricts to those team display names (e.g. today's
    real slate) so a stale/injured player from an unrelated team never gets
    looked up by name collision. Real API status strings only ("Out" /
    "Day-To-Day") -- is_out is the hard signal, short_comment carries
    whatever finer real-world nuance ("questionable"/"probable") exists.
    """
    out: dict[str, dict[str, Any]] = {}
    for team_block in data.get("injuries", []):
        team_name = team_block.get("displayName")
        if teams is not None and team_name not in teams:
            continue
        for inj in team_block.get("injuries", []):
            name = inj.get("athlete", {}).get("displayName")
            if not name:
                continue
            status = (inj.get("status") or "").strip()
            fantasy_abbr = (inj.get("details", {}) or {}).get("fantasyStatus", {}).get("abbreviation", "")
            out[name] = {
                "team": team_name,
                "status": status,
                "is_out": status.lower() in OUT_STATUSES or fantasy_abbr in OUT_FANTASY_ABBR,
                "type": (inj.get("details", {}) or {}).get("type", ""),
                "short_comment": inj.get("shortComment", ""),
                "long_comment": inj.get("longComment", ""),
                "date": inj.get("date", ""),
            }
    return out


def build_usage_vacuum_map(
    injury_map: dict[str, dict[str, Any]], candidate_players: set[str]
) -> dict[str, list[dict[str, Any]]]:
    """candidate_player -> [{injured_teammate, status, note}, ...].

    Real WNBA injury writeups routinely name the specific teammates expected
    to absorb the missing player's minutes/usage (e.g. "Raegan Beers and
    Rayah Marshall are prime candidates to see increased playing time").
    Rather than guess at a minutes-redistribution model, this just checks
    whether each real candidate player's own name is directly named in an
    injured player's own long/short comment -- if the report itself names
    them, that's real, sourced evidence, not an inference. Confirmed
    directly (2026-08-02): correctly recovers Julie Allemand (named as
    Marina Mabrey's beneficiary), Janelle Salaun (Gabby Williams'),
    Diamond Miller (Aaliyah Edwards'), and Isabelle Harrison (Aneesah
    Morrow's) with zero manual hardcoding.
    """
    vacuum: dict[str, list[dict[str, Any]]] = {}
    for injured_name, info in injury_map.items():
        text = f"{info.get('long_comment', '')} {info.get('short_comment', '')}"
        if not text.strip():
            continue
        for player in candidate_players:
            if player == injured_name or player not in text:
                continue
            vacuum.setdefault(player, []).append({
                "injured_teammate": injured_name,
                "status": info["status"],
                "note": info.get("long_comment") or info.get("short_comment"),
            })
    return vacuum
