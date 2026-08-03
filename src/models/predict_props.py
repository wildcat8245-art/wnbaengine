"""Tie it together: real live odds + calibrated models -> a real prop board.

Which engine decides the pick is per-target, chosen by backtest results
(see EMPIRICAL_MC_TARGETS): a real Monte Carlo engine
(empirical_resample_prob_over, bootstrap resampling from each player's own
last 20 real games) for targets where it backtests better, and the fitted
parametric model (Poisson/NegBinom/quantile, trained in
train_baseline_model.py) for targets where the parametric model backtests
better. The parametric model's point estimate and a secondary parametric-
simulation column are always shown for cross-checking regardless of which
engine actually decided the pick.

Usage:
    python -m src.models.predict_props
"""

from __future__ import annotations

import glob
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.data import injuries_client
from src.models.train_baseline_model import (
    FEATURE_COLS, POISSON_TARGETS, NEGATIVE_BINOMIAL_TARGETS,
    load_dataset, train_poisson_model, train_negative_binomial_model, train_quantile_models,
)

MARKET_TO_TARGET = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_threes": "tpm",
}

# Which engine actually drives the pick, per target -- decided by backtest,
# not assumed. backtest_props.py (2026-08-02 run, ~8,900 held-out 2025-2026
# bets) showed the empirical bootstrap engine beats the parametric model for
# points (264K vs 178K synthetic-line bankroll) but loses badly for rebounds
# (13K vs 385K), assists (7.8K vs 21K), and PRA (247K vs 1.24M) -- the
# parametric models for those three have real fixes (NB2 overdispersion fit,
# isotonic recalibration) that the raw bootstrap doesn't get the benefit of.
# Re-run the backtest and update this set if either engine changes.
EMPIRICAL_MC_TARGETS = {"points"}

KELLY_FRACTION = 0.3
EV_THRESHOLD = 1.05
MAX_STAKE_FRACTION = 0.05
BANKROLL = 1000.0


def predict_prob_over(models: dict[float, object], x_row: pd.DataFrame, line: float) -> float:
    """Interpolate the fitted quantile curve to estimate P(stat > line)."""
    qs = sorted(models.keys())
    preds = sorted(model.predict(x_row)[0] for model in models.values())
    # Enforce monotonicity (quantile crossing can happen with independently fit models).
    preds = np.maximum.accumulate(preds)

    if line <= preds[0]:
        return 1 - qs[0] / 2  # line below our lowest modeled quantile: treat as very likely over
    if line >= preds[-1]:
        return (1 - qs[-1]) / 2  # line above our highest modeled quantile: treat as very unlikely over

    cdf_at_line = np.interp(line, preds, qs)
    return 1 - cdf_at_line


def _extended_quantile_grid(qs: list[float], preds: list[float]) -> tuple[list[float], list[float]]:
    """Extend a fitted quantile curve to cover u=0..1 via linear extrapolation
    of the two outer segments, so inverse-transform sampling has somewhere
    real to draw tail values from instead of clamping at the outermost
    fitted quantile (0.1/0.9 here)."""
    slope_low = (preds[1] - preds[0]) / (qs[1] - qs[0])
    val0 = preds[0] - slope_low * qs[0]
    slope_high = (preds[-1] - preds[-2]) / (qs[-1] - qs[-2])
    val1 = preds[-1] + slope_high * (1 - qs[-1])
    return [0.0] + qs + [1.0], [val0] + preds + [val1]


_MC_RNG = np.random.default_rng(42)
MC_SIMULATIONS = 20_000


def parametric_simulation_prob_over(model, x_row: pd.DataFrame, line: float, n_sims: int = MC_SIMULATIONS) -> float:
    """Secondary math check, NOT the primary Monte Carlo engine: draws
    n_sims random samples from the pick's own FITTED distribution (Poisson/
    NegBinom/quantile curve) and reports the empirical fraction that clear
    the line. This is expected to converge to the same number as the
    closed-form survival function -- it verifies the formula, it does not
    add independent information. See empirical_resample_prob_over for the
    real Monte Carlo engine (resamples actual historical games).
    """
    if hasattr(model, "predict_lambda"):
        lam = model.predict_lambda(x_row)[0]
        draws = _MC_RNG.poisson(lam, n_sims)
    elif hasattr(model, "predict_mean") and hasattr(model, "alpha"):
        mu = model.predict_mean(x_row)[0]
        n_param = 1 / model.alpha
        p_param = 1 / (1 + model.alpha * mu)
        draws = _MC_RNG.negative_binomial(n_param, p_param, n_sims)
    else:
        qs = sorted(model.keys())
        preds = list(np.maximum.accumulate(sorted(m.predict(x_row)[0] for m in model.values())))
        ext_qs, ext_preds = _extended_quantile_grid(qs, preds)
        u = _MC_RNG.uniform(0.0, 1.0, n_sims)
        draws = np.clip(np.interp(u, ext_qs, ext_preds), 0, None)
    return float(np.mean(draws > line))


MC_POOL_GAMES = 20
MC_MIN_POOL_GAMES = 8


def build_player_game_pool(features: pd.DataFrame, player: str, target: str, n_recent: int = MC_POOL_GAMES) -> np.ndarray:
    """The player's own real outcomes (raw stat, not per-100) from their
    last n_recent actual games -- the resampling universe for the real
    Monte Carlo engine below. Restricted to recent games (not full career)
    so the pool reflects current role/health, matching the last5/last10
    windows used everywhere else in this codebase."""
    rows = features[features["player_name"] == player].sort_values("game_date")
    return rows[target].tail(n_recent).to_numpy(dtype=float)


def empirical_resample_prob_over(pool: np.ndarray, line: float, n_sims: int = MC_SIMULATIONS,
                                  rng: np.random.Generator | None = None) -> float:
    """The real Monte Carlo engine: bootstrap-resamples n_sims draws (with
    replacement) directly from the player's own real recent game outcomes
    -- not from an assumed parametric shape. Each draw is one real game
    that actually happened, so it carries whatever the true distribution's
    skew/tails/zero-rate actually were that game, with no Poisson/NegBinom/
    quantile assumption in between. Requires MC_MIN_POOL_GAMES real games;
    smaller pools are too noisy to bootstrap meaningfully and the caller
    should fall back to the parametric probability instead."""
    rng = rng or _MC_RNG
    draws = rng.choice(pool, size=n_sims, replace=True)
    return float(np.mean(draws > line))


def predict_point_estimate(model, x_row: pd.DataFrame) -> tuple[float, str]:
    """The model's own real projected value for the stat -- not a simulation.

    Poisson/Negative Binomial models expose their fitted mean directly.
    Quantile models have no single "mean" (only fitted quantiles); the
    0.5 quantile (median) is the closest real equivalent and is reported
    as such rather than mislabeled as a mean.
    """
    if hasattr(model, "predict_lambda"):
        return float(model.predict_lambda(x_row)[0]), "Poisson mean"
    if hasattr(model, "predict_mean"):
        return float(model.predict_mean(x_row)[0]), "NegBinom mean"
    median = model[0.5].predict(x_row)[0]
    return float(median), "quantile median"


def kelly_stake(prob: float, decimal_odds: float, bankroll: float) -> float:
    b = decimal_odds - 1
    q = 1 - prob
    f = (b * prob - q) / b
    f = max(f, 0.0) * KELLY_FRACTION
    f = min(f, MAX_STAKE_FRACTION)
    return f * bankroll


def latest_features_by_player(features: pd.DataFrame) -> pd.DataFrame:
    features = features.sort_values("game_date")
    return features.groupby("player_name").tail(1).set_index("player_name")


# Trend-conflict guard: catches exactly the Rebecca Allen case (real recent
# rate + minutes both trending up, but the model recommends Under anyway).
# Requires the trend to show up in BOTH the per-100 rate AND minutes played,
# not just one noisy stat, before flagging -- a single-game outlier in rate
# alone shouldn't trigger this. Thresholds are deliberately blunt (25%/15%)
# so it only fires on a real, visible trend, not everyday sampling noise.
TREND_RATE_THRESHOLD = 1.25
TREND_MINUTES_THRESHOLD = 1.15


def trend_conflict_flag(side: str, last5: float, last10: float, min5: float, min10: float) -> str:
    rate_ratio = last5 / max(last10, 0.1)
    minutes_ratio = min5 / max(min10, 1.0)

    trending_up = rate_ratio >= TREND_RATE_THRESHOLD and minutes_ratio >= TREND_MINUTES_THRESHOLD
    trending_down = rate_ratio <= 1 / TREND_RATE_THRESHOLD and minutes_ratio <= 1 / TREND_MINUTES_THRESHOLD

    if side == "Under" and trending_up:
        return "CONFLICT: trending UP, model says Under -- STAY AWAY"
    if side == "Over" and trending_down:
        return "CONFLICT: trending DOWN, model says Over -- STAY AWAY"
    return ""


def main() -> int:
    features = load_dataset(Path("data/processed/player_features.csv"))
    latest = latest_features_by_player(features)

    prop_files = sorted(glob.glob("data/raw/daily_props/props_*.csv"))
    if not prop_files:
        print("No daily props snapshot found. Run fetch_daily_props first.")
        return 1
    props = pd.read_csv(prop_files[-1])
    print(f"Using {prop_files[-1]}: {len(props)} prop rows")

    # Real live game-level Vegas total/spread. UPDATE 2026-08-03: the odds
    # key was upgraded to a paid plan with real historical-odds access
    # (archive starts 2022-05-21) -- the spread itself now also feeds the
    # real Garbage-Time & Blowout Risk adjustment below, fit against that
    # real historical data (fetch_historical_spreads.py). The total (and the
    # spread when no blowout-risk adjustment applies) remain informational
    # only, shown in the rationale so the human reading the board can weigh
    # real pace/blowout risk themselves.
    game_line_files = sorted(glob.glob("data/raw/daily_game_lines/game_lines_*.csv"))
    vegas_by_event: dict[str, dict[str, object]] = {}
    player_event: dict[str, str] = {}
    if game_line_files:
        game_lines = pd.read_csv(game_line_files[-1])
        consensus = game_lines.groupby("event_id").agg(
            game_total=("game_total", "median"),
            home_spread=("home_spread", "median"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
        )
        vegas_by_event = consensus.to_dict(orient="index")
        player_event = props.drop_duplicates("player_name").set_index("player_name")["event_id"].to_dict()
        print(f"Vegas game lines: {len(vegas_by_event)} game(s) with a real total/spread snapshot.")
    else:
        print("No game-lines snapshot found -- board will not show Vegas total/spread context.")

    # DISABLED 2026-08-03: the Garbage-Time & Blowout Risk minutes adjustment
    # was backtested (src/backtest/backtest_garbage_time.py) and came back
    # mixed -- a real regression for points (MAE 4.811->4.819) and only
    # tiny, plausibly-noise improvements for rebounds/assists/tpm (e.g.
    # 2.049->2.047). Not strong enough evidence to keep live. Disabled
    # rather than deleted -- garbage_time.py and its backtest are real,
    # tested code kept for reference; do not re-enable without new evidence
    # (a different fit, more data, or a narrower scope) that changes this.
    blowout_model = None

    # Real live injury check -- see injuries_client.py. Restricted to teams
    # actually on today's slate (from the real odds data itself) so a
    # same-named/unrelated player from another team can't collide.
    todays_teams = set(props["home_team"].unique()) | set(props["away_team"].unique())
    try:
        injury_data = injuries_client.fetch_injuries()
        injury_map = injuries_client.build_injury_map(injury_data, teams=todays_teams)
        print(f"Live injury check: {len(injury_map)} players on today's teams have a real injury/status entry.")
    except requests.RequestException as exc:
        print(f"WARNING: live injury check failed ({exc}) -- board will NOT reflect injury status. "
              f"Do not treat any pick below as injury-checked.", file=sys.stderr)
        injury_map = {}

    # Real usage-vacuum check -- see injuries_client.build_usage_vacuum_map.
    # Named directly in the injury reports' own text, not inferred/guessed.
    todays_players = set(props["player_name"].unique())
    vacuum_map = injuries_client.build_usage_vacuum_map(injury_map, todays_players)
    if vacuum_map:
        print(f"Usage-vacuum check: {len(vacuum_map)} player(s) named directly as a teammate's "
              f"minutes/usage beneficiary today: {sorted(vacuum_map.keys())}")

    models_by_target = {}
    for target in set(MARKET_TO_TARGET.values()):
        print(f"Training final {target} model on all available history...")
        if target in POISSON_TARGETS:
            models_by_target[target] = train_poisson_model(features, target)
        elif target in NEGATIVE_BINOMIAL_TARGETS:
            models_by_target[target] = train_negative_binomial_model(features, target)
        else:
            models_by_target[target] = train_quantile_models(features, target)

    # One prob_over per (player, market, line) -- doesn't depend on side/book.
    # Odds DO depend on side and book, so both sides get evaluated against
    # their own real quoted odds and the better side is reported.
    results = []
    grouped = props[props["market"].isin(MARKET_TO_TARGET)].groupby(
        ["player_name", "market", "line", "bookmaker"]
    )
    for (player, market, line, book), group in grouped:
        target = MARKET_TO_TARGET[market]
        if player not in latest.index:
            continue

        over_row = group[group["side"] == "Over"]
        under_row = group[group["side"] == "Under"]
        if over_row.empty or under_row.empty:
            continue  # need both sides' real odds to evaluate fairly

        x_row = latest.loc[[player], FEATURE_COLS].copy()
        if x_row.isna().any(axis=None):
            continue

        # DISABLED 2026-08-03: garbage-time/blowout-risk minutes scaling was
        # here, backtested, and removed -- see the note near blowout_model
        # above for the real numbers. blowout_note kept as an always-empty
        # string since it's still referenced in the rationale/results below.
        blowout_note = ""

        model = models_by_target[target]
        if hasattr(model, "predict_prob_over"):
            model_prob_over = model.predict_prob_over(x_row, line)
        else:
            model_prob_over = predict_prob_over(model, x_row, line)
        projection, projection_type = predict_point_estimate(model, x_row)
        parametric_sim_prob_over = parametric_simulation_prob_over(model, x_row, line)

        # Which engine drives the pick is decided per-target by backtest
        # results (see EMPIRICAL_MC_TARGETS above), not applied blindly to
        # every market. Where it applies, it also falls back to the
        # parametric probability when there isn't enough real game history
        # to bootstrap meaningfully (flagged, not silent).
        if target in EMPIRICAL_MC_TARGETS:
            pool = build_player_game_pool(features, player, target)
            if len(pool) >= MC_MIN_POOL_GAMES:
                prob_over = empirical_resample_prob_over(pool, line)
                mc_source = f"{len(pool)} real games"
            else:
                prob_over = model_prob_over
                mc_source = f"fallback: only {len(pool)} real games (<{MC_MIN_POOL_GAMES}), used model prob"
        else:
            prob_over = model_prob_over
            mc_source = "not used for this market (backtest favors the parametric model, see EMPIRICAL_MC_TARGETS)"

        over_odds = over_row["decimal_odds"].iloc[0]
        under_odds = under_row["decimal_odds"].iloc[0]
        ev_over = prob_over * over_odds
        ev_under = (1 - prob_over) * under_odds

        if ev_over >= ev_under:
            side, prob, odds, ev = "Over", prob_over, over_odds, ev_over
        else:
            side, prob, odds, ev = "Under", 1 - prob_over, under_odds, ev_under

        last5 = x_row[f"{target}_per100_last5"].iloc[0]
        last10 = x_row[f"{target}_per100_last10"].iloc[0]
        min5 = x_row["minutes_last5"].iloc[0]
        min10 = x_row["minutes_last10"].iloc[0]

        flag = trend_conflict_flag(side, last5, last10, min5, min10)

        # Real live injury status (injuries_client.py). A confirmed "Out"
        # (or out-for-season) player categorically shouldn't carry a stake --
        # the model has no way to know this, and books typically void the
        # prop anyway if the player doesn't suit up. Anything softer
        # (Day-To-Day/questionable/probable) is NOT auto-zeroed -- that
        # needs a real judgment call closer to tip-off -- but is always
        # shown, never silently dropped.
        injury = injury_map.get(player)
        if injury and injury["is_out"]:
            flag = f"INJURY: {injury['status']} ({injury['type']}) -- {injury['short_comment']} -- DO NOT BET"
        injury_status = injury["status"] if injury else "no report"
        injury_note = injury["short_comment"] if injury else ""

        # Real usage-vacuum check (injuries_client.build_usage_vacuum_map):
        # a teammate's own injury report names THIS player directly as the
        # minutes/usage beneficiary. More usage generally means more of
        # every counting stat, which structurally argues FOR the Over side
        # and AGAINST the Under side -- confirmed directionally 2026-08-02
        # (Julie Allemand/Diamond Miller Over picks reinforced; Janelle
        # Salaun/Isabelle Harrison Under picks undercut by this exact
        # mechanism). Auto-excludes Under picks the same way a confirmed
        # injury does; Over picks just get an informational note, since the
        # vacuum only strengthens those.
        vacuum_hits = vacuum_map.get(player, [])
        # REVERTED 2026-08-03: this used to also append a quantified real
        # per-36 on/off split (see src/features/on_off_splits.py). Backtested
        # it properly (src/backtest/backtest_on_off_splits.py) and it showed
        # no real out-of-sample predictive validity (in-season split: 0.019
        # correlation, 51.4% sign agreement -- barely above chance, even
        # after fixing the original multi-year-gap methodology issue).
        # Reverted to the qualitative-only flag below, which IS real and
        # sourced (the injury report's own text naming a beneficiary), just
        # not a validated number. Do not re-add the quantified version
        # without a genuinely new, tested hypothesis for why it would work --
        # this exact approach was tried and failed validation twice.
        vacuum_note = ""
        if vacuum_hits:
            hit = vacuum_hits[0]
            vacuum_note = (
                f"{hit['injured_teammate']} ({hit['status']}) named directly as this player's "
                f"minutes/usage beneficiary -- {hit['note']}"
            )

            if side == "Under" and not flag:
                flag = f"USAGE VACUUM: {vacuum_note} -- more usage argues against Under -- excluded"

        # Real live Vegas game total/spread -- informational only (see the
        # NOTE where vegas_by_event is built above for why this doesn't feed
        # prob_over/EV/staking). Gives the human reading the board a real
        # market read on pace/blowout risk to weigh alongside the model.
        vegas_note = ""
        event_id = player_event.get(player)
        game_line = vegas_by_event.get(event_id) if event_id else None
        if game_line:
            total = game_line.get("game_total")
            spread = game_line.get("home_spread")
            home_team = game_line.get("home_team")
            parts = []
            if pd.notna(total):
                parts.append(f"total {total:.1f}")
            if pd.notna(spread) and home_team:
                parts.append(f"{home_team} {spread:+.1f}")
            if parts:
                vegas_note = "Vegas: " + ", ".join(parts) + "."

        # A flagged pick is a stay-away by definition -- no stake regardless
        # of what the raw EV says, since the whole point of the flag is that
        # the model/trend/injury status actively disagree with betting it.
        stake = 0.0 if flag else (kelly_stake(prob, odds, BANKROLL) if ev > EV_THRESHOLD else 0.0)

        # Real raw outcomes context -- shown regardless of which engine
        # decided the pick, so every row is auditable against the player's
        # own actual recent games, not just the model's internal number.
        context_pool = build_player_game_pool(features, player, target)
        pool_mean = float(np.mean(context_pool)) if len(context_pool) else None
        diff_vs_line = round(projection - line, 2)

        # NOTE: projection is the model's median (quantile targets) or mean
        # (Poisson/NegBinom targets) -- a single point summary. The actual
        # pick is decided by prob_over (the full probability mass above the
        # line), which can legitimately point the OPPOSITE direction from
        # the point summary for a skewed distribution (e.g. median can sit
        # below a line while most probability mass still sits above it).
        # Do NOT infer "model favors X" from the sign of diff_vs_line alone
        # -- state both numbers as facts and let prob_over be the one true
        # explanation for which side was picked.
        engine_desc = (
            f"empirical Monte Carlo ({mc_source})"
            if target in EMPIRICAL_MC_TARGETS and "fallback" not in mc_source
            else "parametric model" + (f" (MC {mc_source})" if target in EMPIRICAL_MC_TARGETS else "")
        )
        proj_vs_mean_conflict = (
            pool_mean is not None
            and ((projection > line) != (pool_mean > line))
        )
        rationale = (
            f"{projection_type} (point estimate) = {projection:.2f} vs line {line} (diff {diff_vs_line:+.2f}). "
            + (f"Real last-20-game raw average = {pool_mean:.2f}. " if pool_mean is not None else "")
            + (
                "NOTE: the point-estimate projection and the real recent average sit on OPPOSITE "
                "sides of the line -- expected for a skewed distribution where median/mean != "
                "the probability split, not an error, but worth a second look. "
                if proj_vs_mean_conflict else ""
            )
            + f"Last-5 real per-100 rate {last5:.2f} vs last-10 {last10:.2f}, minutes {min5:.1f} vs {min10:.1f}. "
            f"Injury status: {injury_status}"
            f"{' -- ' + injury_note if injury_note else ''}. "
            + (f"Usage vacuum: {vacuum_note}. " if vacuum_note else "")
            + (f"{vegas_note} " if vegas_note else "")
            + (blowout_note if blowout_note else "")
            + f"DECISION: picked {side} because the actual probability of clearing {line} was "
            f"{prob_over*100:.1f}% Over / {(1-prob_over)*100:.1f}% Under, computed by the {engine_desc}. "
            f"EV {ev:.3f} at {odds:.2f} decimal odds on {book}."
            f"{' *** ' + flag + ' ***' if flag else ''}"
        )

        results.append({
            "player": player,
            "market": target,
            "market_line": line,
            "book": book,
            "side": side,
            "decimal_odds": odds,
            "system_projection": round(projection, 2),
            "projection_type": projection_type,
            "diff_vs_line": diff_vs_line,
            "real_last20_mean": round(pool_mean, 2) if pool_mean is not None else "N/A",
            "injury_status": injury_status,
            "injury_note": injury_note,
            "usage_vacuum": vacuum_note,
            "vegas_note": vegas_note,
            "blowout_adjustment": blowout_note,
            "monte_carlo_over_pct": f"{prob_over * 100:.1f}%",
            "monte_carlo_pool": mc_source,
            "pick_pct": f"{prob * 100:.1f}% {side}",
            "model_prob_over_pct": f"{model_prob_over * 100:.1f}%",
            "parametric_sim_over_pct": f"{parametric_sim_prob_over * 100:.1f}%",
            "ev": round(ev, 3),
            "kelly_stake": round(stake, 2),
            "flag": flag,
            "rationale": rationale,
            # Supporting context -- shown for every pick, not just when asked.
            # Generic column names (not f"{target}_...") so every row uses the
            # same columns regardless of market -- otherwise the board would
            # end up sparse/inconsistent across points/rebounds/assists rows.
            "stat_per100_last5": round(last5, 2),
            "stat_per100_last10": round(last10, 2),
            "minutes_last5_vs_last10": f"{min5:.1f} vs {min10:.1f}",
        })

    board = pd.DataFrame(results).sort_values("ev", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    print(board.to_string(index=False))

    out_path = Path(f"data/processed/predictions_{date.today().isoformat()}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(out_path, index=False)
    print(f"\nFull board, every column (system_projection, diff_vs_line, real_last20_mean, "
          f"rationale, etc.), written to {out_path} -- open in Excel/Sheets for the complete layout.")

    flagged = board[board["flag"] != ""]
    positive_ev = board[board["kelly_stake"] > 0]
    print(f"\n{len(positive_ev)} bets clear the EV>{EV_THRESHOLD} threshold out of {len(board)} evaluated.")

    print(f"\nFull rationale for the {min(20, len(positive_ev))} highest-EV recommended bets:")
    for _, row in positive_ev.head(20).iterrows():
        print(f"\n- {row['player']} | {row['market']} {row['side']} {row['market_line']} "
              f"@ {row['book']} ({row['decimal_odds']}) | EV {row['ev']} | stake ${row['kelly_stake']}")
        print(f"  {row['rationale']}")
    if not flagged.empty:
        print(f"\n{len(flagged)} pick(s) had a high EV but were caught by the trend-conflict guard "
              f"(stake forced to 0, not recommended):")
        print(flagged[["player", "market", "side", "market_line", "ev", "stat_per100_last5",
                        "stat_per100_last10", "minutes_last5_vs_last10"]].drop_duplicates(
                            subset=["player", "market", "side", "market_line"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
