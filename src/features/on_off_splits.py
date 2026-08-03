"""Real teammate on/off splits: a player's own real rate in games a specific
teammate did or did not play, computed directly from historical box scores
-- no new API pulls needed. Replaces the purely qualitative usage-vacuum
flag (injuries_client.build_usage_vacuum_map only checks whether an injury
report's TEXT names a teammate) with an actual measured number wherever real
sample size supports it.

Scope note (deliberate, not an oversight): this is the coarser GAME-level
on/off split -- did the teammate play at all in this game, not real
on-court-stint overlap within the game. The finer version (true minute-by-
minute lineup overlap) needs real play-by-play for thousands of historical
games -- confirmed buildable (real substitution events exist per game,
see SESSION_NOTES), but a much larger, separate undertaking not done here.
Full 5-man lineup-combination splits are deliberately NOT attempted: most
lineup combinations only ever share the floor for a handful of real
possessions across a whole WNBA season, too small a sample to report as a
reliable number rather than noise dressed up as precision.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MIN_OUT_GAMES = 5  # minimum real "teammate did not play" games before reporting a split


def load_played_minutes(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path, dtype={"game_id": str, "player_id": str})
    df["game_date"] = pd.to_datetime(df["game_date"], format="%Y-%m-%d")
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    for stat in ("points", "rebounds", "assists"):
        df[stat] = pd.to_numeric(df[stat], errors="coerce")
    return df


def compute_on_off_split(
    df: pd.DataFrame,
    team: str,
    teammate_out_name: str,
    candidate_names: set[str],
    season_start: pd.Timestamp | None = None,
    min_out_games: int = MIN_OUT_GAMES,
) -> dict[str, dict]:
    """For each candidate teammate on `team`, their real per-36-minutes rate
    (points/rebounds/assists) in real games `teammate_out_name` did NOT play
    vs DID play. Per-36 rather than per-100-possessions here so this module
    stays self-contained (no dependency on the team-possessions pipeline).

    Returns, per candidate name: games_without / games_with (real counts),
    rate_without / rate_with (dicts of real per-36 rates, or None),
    pct_change (real % difference per stat, or None), and insufficient_data
    (True whenever games_without or games_with is below min_out_games --
    never fabricates a split from too small a real sample).
    """
    team_rows = df[df["team"] == team]
    if season_start is not None:
        team_rows = team_rows[team_rows["game_date"] >= season_start]

    all_games = set(team_rows["game_id"])
    out_played_games = set(
        team_rows.loc[
            (team_rows["player_name"] == teammate_out_name) & (team_rows["minutes"].fillna(0) > 0),
            "game_id",
        ]
    )
    games_without_teammate = all_games - out_played_games
    games_with_teammate = out_played_games

    def per36(rows: pd.DataFrame, stat: str) -> float:
        total_minutes = rows["minutes"].sum()
        return float(rows[stat].sum() / total_minutes * 36) if total_minutes > 0 else 0.0

    results: dict[str, dict] = {}
    for name in candidate_names:
        if name == teammate_out_name:
            continue
        player_rows = team_rows[(team_rows["player_name"] == name) & (team_rows["minutes"].fillna(0) > 0)]
        without = player_rows[player_rows["game_id"].isin(games_without_teammate)]
        with_ = player_rows[player_rows["game_id"].isin(games_with_teammate)]
        n_without, n_with = len(without), len(with_)

        if n_without < min_out_games or n_with < min_out_games:
            results[name] = {
                "games_without": n_without, "games_with": n_with,
                "rate_without": None, "rate_with": None, "pct_change": None,
                "insufficient_data": True,
            }
            continue

        rate_without = {stat: per36(without, stat) for stat in ("points", "rebounds", "assists")}
        rate_with = {stat: per36(with_, stat) for stat in ("points", "rebounds", "assists")}
        pct_change = {
            stat: ((rate_without[stat] - rate_with[stat]) / rate_with[stat] * 100 if rate_with[stat] > 0 else None)
            for stat in ("points", "rebounds", "assists")
        }
        results[name] = {
            "games_without": n_without, "games_with": n_with,
            "rate_without": rate_without, "rate_with": rate_with, "pct_change": pct_change,
            "insufficient_data": False,
        }
    return results
