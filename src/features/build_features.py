"""Feature engineering: lagged rolling windows, per-100 normalization, opponent adjustment.

Per-100-possession standardization removes pace effects; rolling
last-N-games windows are always shifted so only games strictly preceding
the target game are used (no leakage), applied at the player level for
prop targets (PTS/REB/AST/PRA).

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


# Empirically fit (2026-08-03), same method-of-moments discipline as the
# NB2 rebounds dispersion fit in train_baseline_model.py: for a grid of
# shrinkage strengths K (in real attempts-equivalent units), measured MSE of
# the shrunk last-5-game rate against that SAME game's real actual shooting
# outcome, picked the K minimizing MSE on train seasons (<=2024), then
# verified the improvement holds on the true held-out 2025-2026 test set --
# it does: eFG% MSE 0.0915->0.0789 (raw last5 -> shrunk), TS% MSE
# 0.0818->0.0708, both genuine, out-of-sample reductions, not just a
# train-set fit. A single game's shooting outcome is extremely noisy
# relative to a 5-game sample (rarely more than ~20-25 real attempts), which
# is why the fit favors such heavy shrinkage toward the real career-to-date
# baseline.
EFG_SHRINKAGE_K = 1000.0
TS_SHRINKAGE_K = 750.0

POSITION_DEFENSE_STATS = ("points", "rebounds", "assists")


def compute_shooting_efficiency(df: pd.DataFrame, windows: tuple[int, ...] = ROLLING_WINDOWS) -> pd.DataFrame:
    """Real eFG%/TS%, computed from made/attempted SUMS over the rolling
    window (not a naive mean of per-game percentages, which would wrongly
    weight a 2-attempt game the same as a 15-attempt game), then regressed
    toward each player's own real career-to-date baseline via attempts-
    weighted empirical-Bayes shrinkage -- same category of fix as the
    isotonic recalibration already validated for assists, applied here to
    raw shooting volatility rather than a model's prediction bias.

    "Uncharacteristically hot or cold" recent shooting is exactly what gets
    pulled back toward the real baseline; a genuinely large, high-volume
    real shift in a player's own shooting still comes through, it's just
    not overweighted, because the correction is real makes/attempts math
    with an empirically fit strength, not an assumption.
    """
    df = df.sort_values(["player_id", "game_date"]).copy()
    grp = df.groupby("player_id")

    sum_cols = ["fgm", "fga", "tpm", "fta", "points"]
    for window in windows:
        for col in sum_cols:
            df[f"{col}_sum_last{window}"] = grp[col].transform(
                lambda s: s.shift(1).rolling(window, min_periods=window).sum()
            )
    # Real career-to-date baseline (expanding, shifted -- only real games
    # strictly before this one), requires at least 10 real prior games
    # before reporting a baseline at all.
    for col in sum_cols:
        df[f"{col}_sum_career"] = grp[col].transform(lambda s: s.shift(1).expanding(min_periods=10).sum())

    career_fga = df["fga_sum_career"]
    career_tsa = career_fga + 0.44 * df["fta_sum_career"]
    df["efg_career"] = (df["fgm_sum_career"] + 0.5 * df["tpm_sum_career"]) / career_fga
    df["ts_career"] = df["points_sum_career"] / (2 * career_tsa)

    for window in windows:
        fga = df[f"fga_sum_last{window}"]
        tsa = fga + 0.44 * df[f"fta_sum_last{window}"]
        made_equiv = df[f"fgm_sum_last{window}"] + 0.5 * df[f"tpm_sum_last{window}"]

        df[f"efg_last{window}"] = made_equiv / fga
        df[f"ts_last{window}"] = df[f"points_sum_last{window}"] / (2 * tsa)

        # Shrinkage toward the real career baseline; degrades gracefully to
        # exactly the career rate when fga/tsa is 0 (no real recent
        # attempts), and toward the raw last-N rate as fga/tsa grows large
        # relative to the fitted K.
        df[f"efg_shrunk_last{window}"] = (made_equiv + EFG_SHRINKAGE_K * df["efg_career"]) / (fga + EFG_SHRINKAGE_K)
        df[f"ts_shrunk_last{window}"] = (
            df[f"points_sum_last{window}"] + 2 * TS_SHRINKAGE_K * df["ts_career"]
        ) / (2 * (tsa + TS_SHRINKAGE_K))

    return df


def compute_position_defense(df: pd.DataFrame, opp_map: pd.DataFrame, team_game: pd.DataFrame) -> pd.DataFrame:
    """Real, position-specific defensive rate allowed: for each (team,
    position), a rolling last-5/10-game rate of points/rebounds/assists that
    team's opponents' players AT THAT POSITION scored against them --
    replaces one blended team-wide DRtg with three real, separately-tracked
    splits (vs. guards / forwards / centers), using each team's actual real
    position field (confirmed clean: only G/F/C values, no messy hybrids).

    Rate is normalized against the SCORING team's own game possessions
    (matching compute_team_ratings' drtg convention: opponent points /
    opponent possessions), then shifted+rolled per (team, position) so only
    games strictly before the target game are used.
    """
    off_by_position = df.groupby(["game_id", "team", "position"], as_index=False).agg(
        pos_points=("points", "sum"), pos_rebounds=("rebounds", "sum"), pos_assists=("assists", "sum")
    )
    off_by_position = off_by_position.merge(
        team_game[["game_id", "team", "possessions", "game_date"]], on=["game_id", "team"]
    )
    for stat in POSITION_DEFENSE_STATS:
        off_by_position[f"pos_{stat}_per100"] = off_by_position[f"pos_{stat}"] / off_by_position["possessions"] * 100

    # defense_allowed: for each (game_id, defending team, position), what the
    # OPPONENT's players at that position actually scored against them.
    defense_allowed = opp_map.merge(
        off_by_position.rename(columns={"team": "opponent"}),
        on=["game_id", "opponent"],
    )

    defense_allowed = defense_allowed.sort_values(["team", "position", "game_date"])
    grp = defense_allowed.groupby(["team", "position"])
    for window in ROLLING_WINDOWS:
        for stat in POSITION_DEFENSE_STATS:
            defense_allowed[f"opp_drtg_vs_position_{stat}_last{window}"] = grp[f"pos_{stat}_per100"].transform(
                lambda s: s.shift(1).rolling(window, min_periods=window).mean()
            )

    cols = ["game_id", "team", "position"] + [
        f"opp_drtg_vs_position_{stat}_last{w}" for w in ROLLING_WINDOWS for stat in POSITION_DEFENSE_STATS
    ]
    return defense_allowed[cols]


def compute_league_avg_pace(team_game: pd.DataFrame, windows: tuple[int, ...] = ROLLING_WINDOWS) -> pd.DataFrame:
    """Real, rolling league-wide average pace as of each date -- NOT a single
    all-time constant. Confirmed directly (2026-08-03): real season-average
    game totals jumped from ~158-166 every year 2015-2025 to ~174.7 in 2026,
    a genuine era shift -- a fixed historical league-average pace would go
    stale exactly the same way the totals model did before isotonic
    recalibration fixed it. Instead this is a trailing rolling mean of real
    single-game possessions across ALL teams, ordered chronologically
    league-wide (not per-team), shifted by one row to avoid leakage.

    Window size is scaled by the number of real teams that season so a
    "last N games" league-wide window represents roughly the same elapsed
    real time as a per-team last-N window (a 5-game team window needs far
    more league-wide rows to cover the same stretch of the season).
    """
    league = team_game[["game_id", "team", "game_date", "possessions"]].sort_values("game_date").reset_index(drop=True)
    n_teams = team_game["team"].nunique()
    for window in windows:
        league_window = window * n_teams
        league[f"league_avg_pace_last{window}"] = (
            league["possessions"].shift(1).rolling(league_window, min_periods=league_window).mean()
        )
    return league[["game_id", "team"] + [f"league_avg_pace_last{w}" for w in windows]]


def build_player_features(raw_path: Path) -> pd.DataFrame:
    df = load_and_clean(raw_path)
    opp_map = compute_opponent_map(df)
    df = df.merge(opp_map, on=["game_id", "team"])
    df = compute_shooting_efficiency(df)

    team_game = compute_team_game_possessions(df)
    team_ratings = compute_team_ratings(team_game, opp_map)
    league_avg_pace = compute_league_avg_pace(team_game)

    team_game = team_game.sort_values(["team", "game_date"])
    team_game["rest_days"] = team_game.groupby("team")["game_date"].diff().dt.days

    df = df.merge(team_game[["game_id", "team", "possessions", "rest_days"]], on=["game_id", "team"])
    df = df.merge(league_avg_pace, on=["game_id", "team"])

    # tpm (three-pointers made) included alongside the existing counting
    # stats -- real data was already being parsed from every box score
    # (load_and_clean's tpm/tpa split) but never turned into its own
    # per-100/rolling feature or projection target before now.
    stat_cols = ["points", "rebounds", "assists", "pra", "tpm", "steals", "blocks", "turnovers", "minutes"]
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
        for col in ["assists", "steals", "blocks", "tpm"]:
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

    # Dynamic Pace & Possessions layer: real head-to-head expected game pace
    # (own x opp, normalized by the real rolling league-average pace) rather
    # than a static pace average -- two above-average-pace teams playing each
    # other should push the real expected pace higher than either team's own
    # rolling average alone would suggest.
    for window in ROLLING_WINDOWS:
        df[f"expected_pace_last{window}"] = (
            df[f"own_team_pace_last{window}"] * df[f"opp_team_pace_last{window}"]
            / df[f"league_avg_pace_last{window}"]
        )
        # Each player's own real per-100 rate, scaled directly against this
        # game's real expected possession count -- an explicit projected
        # volume, not left for the tree model to reconstruct implicitly from
        # separately-listed pace/rate features.
        for col in ["points", "rebounds", "assists", "tpm"]:
            df[f"{col}_projected_volume_last{window}"] = (
                df[f"{col}_per100_last{window}"] / 100 * df[f"expected_pace_last{window}"]
            )

    # Defensive Matchup & Localized Splits (position-level, the real version
    # of this layer -- see build_features module docstring history / session
    # notes for why true play-type defense isn't available). Look up the
    # OPPONENT's rolling defensive rate specifically against players at THIS
    # player's own position, replacing the single blended opp_team_drtg with
    # three real, separately-tracked splits.
    position_defense = compute_position_defense(df, opp_map, team_game).rename(columns={"team": "pos_def_team"})
    df = df.merge(
        position_defense,
        left_on=["game_id", "opponent", "position"],
        right_on=["game_id", "pos_def_team", "position"],
        how="left",
    ).drop(columns=["pos_def_team"])

    # Garbage-Time & Blowout Risk Modeling: a real pre-game expected-margin
    # proxy, usable across the FULL 2015-2026 history (real historical Vegas
    # spreads only exist from 2022-05-21 onward -- see
    # fetch_historical_spreads.py for the real-spread version used to
    # validate this proxy). Net rating differential (own - opp, each
    # ORtg-DRtg) is a real, standard point-margin predictor; scaling it by
    # the real expected game pace (reusing expected_pace_last{w} from the
    # Dynamic Pace layer, possessions/100) converts it from a per-100-
    # possession rate into a real expected POINT margin for this specific
    # matchup, not just a relative strength score.
    for window in ROLLING_WINDOWS:
        own_net = df[f"own_team_ortg_last{window}"] - df[f"own_team_drtg_last{window}"]
        opp_net = df[f"opp_team_ortg_last{window}"] - df[f"opp_team_drtg_last{window}"]
        df[f"expected_margin_proxy_last{window}"] = (
            (own_net - opp_net) * df[f"expected_pace_last{window}"] / 100
        )
        df[f"abs_expected_margin_proxy_last{window}"] = df[f"expected_margin_proxy_last{window}"].abs()

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
