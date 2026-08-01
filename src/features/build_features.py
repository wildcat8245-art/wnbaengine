"""Feature engineering: lagged rolling windows, per-100 normalization, opponent adjustment.

Mirrors the paper's approach (per-100-possession standardization removes
pace effects; rolling last-N-games windows, always shifted so only games
strictly preceding the target game are used) but applied at the player
level for prop targets (PTS/REB/AST/PRA) instead of team win/loss.

Possession estimate uses the standard single-team approximation
(FGA - OREB + TOV + 0.44*FTA). This is a real, if imperfect, formula
choice: computing it exactly requires both teams' rebounding splits,
which the box score does support, but the single-team approximation is
the widely used standard and is accurate to within a few percent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROLLING_WINDOWS = (5, 10)
MIN_TEAM_GAMES_TO_KEEP = 20  # filters out one-off All-Star "teams" (e.g. "Team Wilson")


def _split_made_attempted(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = series.fillna("0-0").str.split("-", n=1, expand=True)
    made = pd.to_numeric(parts[0], errors="coerce")
    attempted = pd.to_numeric(parts[1], errors="coerce")
    return made, attempted


def load_and_clean(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path, dtype={"game_id": str, "player_id": str})
    df["game_date"] = pd.to_datetime(df["game_date"], format="%Y-%m-%d")

    df["fgm"], df["fga"] = _split_made_attempted(df["fieldGoalsMade-fieldGoalsAttempted"])
    df["tpm"], df["tpa"] = _split_made_attempted(df["threePointFieldGoalsMade-threePointFieldGoalsAttempted"])
    df["ftm"], df["fta"] = _split_made_attempted(df["freeThrowsMade-freeThrowsAttempted"])

    numeric_cols = [
        "minutes", "points", "rebounds", "assists", "turnovers", "steals",
        "blocks", "offensiveRebounds", "defensiveRebounds", "fouls",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["did_not_play"] = df["did_not_play"].astype(bool)
    df["is_home"] = df["home_away"].eq("home").astype(int)
    played = df["did_not_play"].eq(False) & df["minutes"].fillna(0).gt(0)
    df = df[played].copy()

    # Drop one-off exhibition rosters (All-Star draft teams etc.) — real
    # teams appear in far more than MIN_TEAM_GAMES_TO_KEEP games.
    team_game_counts = df.groupby("team")["game_id"].nunique()
    real_teams = team_game_counts[team_game_counts >= MIN_TEAM_GAMES_TO_KEEP].index
    df = df[df["team"].isin(real_teams)].copy()

    df["pra"] = df["points"] + df["rebounds"] + df["assists"]
    return df


def compute_opponent_map(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team) -> opponent team abbreviation.

    Games with anything other than exactly two distinct real teams are
    dropped rather than guessed at.
    """
    teams_per_game = df.groupby("game_id")["team"].unique()
    valid_games = teams_per_game[teams_per_game.apply(len) == 2]

    rows = []
    for game_id, teams in valid_games.items():
        a, b = teams
        rows.append({"game_id": game_id, "team": a, "opponent": b})
        rows.append({"game_id": game_id, "team": b, "opponent": a})
    return pd.DataFrame(rows)


def compute_team_game_possessions(df: pd.DataFrame) -> pd.DataFrame:
    """Team-level totals and possession estimate for each (game_id, team)."""
    team_game = df.groupby(["game_id", "game_date", "team"], as_index=False).agg(
        team_points=("points", "sum"),
        team_fga=("fga", "sum"),
        team_oreb=("offensiveRebounds", "sum"),
        team_tov=("turnovers", "sum"),
        team_fta=("fta", "sum"),
    )
    team_game["possessions"] = (
        team_game["team_fga"] - team_game["team_oreb"] + team_game["team_tov"] + 0.44 * team_game["team_fta"]
    )
    return team_game


def compute_team_ratings(team_game: pd.DataFrame, opp_map: pd.DataFrame) -> pd.DataFrame:
    """Attach each team's own game ORtg/DRtg/pace, then lagged rolling means.

    DRtg here uses the opponent's own possession estimate for that game as
    the defensive-possessions-faced proxy (the two teams' single-game pace
    estimates are close but not identical due to rebounding differences;
    using the opponent's own figure is the more correct side of that).
    """
    merged = team_game.merge(opp_map, on=["game_id", "team"])
    merged = merged.merge(
        team_game[["game_id", "team", "team_points", "possessions"]].rename(
            columns={"team": "opponent", "team_points": "opp_points", "possessions": "opp_possessions"}
        ),
        on=["game_id", "opponent"],
    )

    merged["ortg"] = merged["team_points"] / merged["possessions"] * 100
    merged["drtg"] = merged["opp_points"] / merged["opp_possessions"] * 100
    merged["pace"] = merged["possessions"]

    merged = merged.sort_values(["team", "game_date"])
    for window in ROLLING_WINDOWS:
        grp = merged.groupby("team")
        merged[f"team_ortg_last{window}"] = grp["ortg"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).mean()
        )
        merged[f"team_drtg_last{window}"] = grp["drtg"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).mean()
        )
        merged[f"team_pace_last{window}"] = grp["pace"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).mean()
        )

    return merged[["game_id", "team", "opponent", "possessions"] + [
        c for c in merged.columns if c.startswith("team_ortg_last")
        or c.startswith("team_drtg_last") or c.startswith("team_pace_last")
    ]]


def build_player_features(raw_path: Path) -> pd.DataFrame:
    df = load_and_clean(raw_path)
    opp_map = compute_opponent_map(df)
    df = df.merge(opp_map, on=["game_id", "team"])

    team_game = compute_team_game_possessions(df)
    team_ratings = compute_team_ratings(team_game, opp_map)

    team_game = team_game.sort_values(["team", "game_date"])
    team_game["rest_days"] = team_game.groupby("team")["game_date"].diff().dt.days

    df = df.merge(team_game[["game_id", "team", "possessions", "rest_days"]], on=["game_id", "team"])

    stat_cols = ["points", "rebounds", "assists", "pra", "steals", "blocks", "turnovers", "minutes"]
    for col in stat_cols:
        df[f"{col}_per100"] = df[col] / df["possessions"] * 100

    df = df.sort_values(["player_id", "game_date"])
    grp = df.groupby("player_id")
    for window in ROLLING_WINDOWS:
        for col in stat_cols:
            df[f"{col}_per100_last{window}"] = grp[f"{col}_per100"].transform(
                lambda s: s.shift(1).rolling(window, min_periods=window).mean()
            )
        df[f"minutes_last{window}"] = grp["minutes"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).mean()
        )
        # Explicit zero-rate feature: gives the low-quantile model a direct
        # signal instead of forcing it to infer zero-proneness indirectly
        # from minutes (which it was failing to do -- see assists fix).
        for col in ["assists", "steals", "blocks"]:
            df[f"{col}_zero_rate_last{window}"] = grp[col].transform(
                lambda s: s.shift(1).eq(0).rolling(window, min_periods=window).mean()
            )

    # Attach the player's own team's recent form and the opponent's recent
    # defensive form (both lagged, i.e. entering this game).
    df = df.merge(
        team_ratings.add_prefix("own_"),
        left_on=["game_id", "team"],
        right_on=["own_game_id", "own_team"],
        how="left",
    )
    df = df.merge(
        team_ratings.add_prefix("opp_"),
        left_on=["game_id", "opponent"],
        right_on=["opp_game_id", "opp_team"],
        how="left",
    )

    return df


def main() -> int:
    raw_path = Path("data/raw/player_boxscores_historical.csv")
    out_path = Path("data/processed/player_features.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    features = build_player_features(raw_path)
    features.to_csv(out_path, index=False)
    print(f"Wrote {len(features)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
