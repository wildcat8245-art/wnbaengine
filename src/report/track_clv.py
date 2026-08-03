"""Real-Time Execution & Feedback Logging: Closing Line Value (CLV) tracking.

CLV -- whether the real odds we actually bet at were better than the real
CLOSING line right before tip-off -- is a standard, real sports-betting
skill signal, and one that doesn't require waiting for the game outcome to
resolve (unlike win rate). Only possible now because of the paid
ODDS_API_KEY upgrade found this session: queries a real historical odds
snapshot at each game's own real commence_time to get the real closing
line, then compares it to what we actually recorded at pick-time in the
saved predictions_*.csv.

Only processes saved boards whose games have already been played (real
commence_time in the past) -- there is no closing line for a game that
hasn't happened yet.

Usage:
    python -m src.report.track_clv
"""

from __future__ import annotations

import glob
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.data.odds_client import _api_key, BASE_URL as LIVE_BASE_URL

HISTORICAL_BASE_URL = "https://api.the-odds-api.com/v4/historical/sports/basketball_wnba"
MARKET_TO_ODDS_KEY = {
    "points": "player_points", "rebounds": "player_rebounds",
    "assists": "player_assists", "tpm": "player_threes",
}
OUTPUT_PATH = Path("data/processed/clv_report.csv")


def _date_from_predictions_filename(path: Path) -> str | None:
    m = re.search(r"predictions_(\d{4}-\d{2}-\d{2})\.csv$", path.name)
    return m.group(1) if m else None


def fetch_closing_odds(event_id: str, commence_time: str, markets: list[str], session: requests.Session) -> dict:
    """Real closing-line snapshot: query the historical odds archive AT the
    game's own real commence_time (the definition of "closing line" -- the
    last real market price before tip-off)."""
    resp = session.get(
        f"{HISTORICAL_BASE_URL}/events/{event_id}/odds",
        params={"apiKey": _api_key(), "regions": "us", "markets": ",".join(markets), "date": commence_time},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json().get("data") or {}


def build_closing_lookup(closing_data: dict) -> dict[tuple[str, str, str, float], list[float]]:
    """(player, market, side, line) -> [real closing decimal odds across books]."""
    lookup: dict[tuple, list[float]] = {}
    for bookmaker in closing_data.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            for outcome in market.get("outcomes", []):
                key = (outcome.get("description"), market.get("key"), outcome.get("name"), outcome.get("point"))
                lookup.setdefault(key, []).append(outcome.get("price"))
    return lookup


def track_board(predictions_path: Path, board_date: str, session: requests.Session) -> pd.DataFrame:
    df = pd.read_csv(predictions_path)
    recommended = df[df["kelly_stake"] > 0].drop_duplicates(subset=["player", "market", "market_line", "side"])
    if recommended.empty:
        return pd.DataFrame()

    props_path = Path(f"data/raw/daily_props/props_{board_date}.csv")
    if not props_path.exists():
        print(f"  no daily props snapshot for {board_date}, skipping", file=sys.stderr)
        return pd.DataFrame()
    props = pd.read_csv(props_path)

    player_event = props.drop_duplicates("player_name").set_index("player_name")[
        ["event_id", "commence_time"]
    ].to_dict(orient="index")

    now = datetime.now(timezone.utc)
    rows = []
    events_needed: dict[str, str] = {}
    for _, row in recommended.iterrows():
        info = player_event.get(row["player"])
        if not info:
            continue
        commence = datetime.fromisoformat(info["commence_time"].replace("Z", "+00:00"))
        if commence >= now:
            continue  # game hasn't happened yet -- no real closing line exists
        events_needed[info["event_id"]] = info["commence_time"]

    closing_by_event: dict[str, dict] = {}
    for event_id, commence_time in events_needed.items():
        markets_for_event = list(MARKET_TO_ODDS_KEY.values())
        try:
            data = fetch_closing_odds(event_id, commence_time, markets_for_event, session)
        except requests.RequestException as exc:
            print(f"  closing-odds fetch failed for event {event_id}: {exc}", file=sys.stderr)
            continue
        closing_by_event[event_id] = build_closing_lookup(data)

    for _, row in recommended.iterrows():
        info = player_event.get(row["player"])
        if not info or info["event_id"] not in closing_by_event:
            continue
        odds_key = MARKET_TO_ODDS_KEY.get(row["market"])
        if odds_key is None:
            continue
        lookup = closing_by_event[info["event_id"]]
        key = (row["player"], odds_key, row["side"], float(row["market_line"]))
        closing_prices = lookup.get(key)
        if not closing_prices:
            continue
        closing_median = sorted(closing_prices)[len(closing_prices) // 2]
        our_odds = float(row["decimal_odds"])
        clv_pct = (our_odds / closing_median - 1) * 100

        rows.append({
            "board_date": board_date, "player": row["player"], "market": row["market"],
            "line": row["market_line"], "side": row["side"], "our_odds": our_odds,
            "closing_odds_median": closing_median, "n_closing_books": len(closing_prices),
            "clv_pct": round(clv_pct, 2), "beat_close": clv_pct > 0,
        })

    return pd.DataFrame(rows)


def main() -> int:
    predictions_files = sorted(glob.glob("data/processed/predictions_*.csv"))
    if not predictions_files:
        print("No saved predictions files found.")
        return 1

    all_results = []
    with requests.Session() as session:
        for path_str in predictions_files:
            path = Path(path_str)
            board_date = _date_from_predictions_filename(path)
            if board_date is None:
                continue
            print(f"Processing {path.name}...", file=sys.stderr)
            result = track_board(path, board_date, session)
            if not result.empty:
                all_results.append(result)

    if not all_results:
        print("No CLV-eligible bets found (no saved recommended bets from already-played games).")
        return 0

    report = pd.concat(all_results, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(report)} CLV-tracked bets to {OUTPUT_PATH}")
    print(f"Real average CLV: {report['clv_pct'].mean():+.2f}%")
    print(f"Beat the real closing line: {report['beat_close'].mean():.1%} of bets")
    print("\nBy market:")
    print(report.groupby("market")["clv_pct"].agg(["mean", "count"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
