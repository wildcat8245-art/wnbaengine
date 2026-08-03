"""Baseline calibrated player-prop model: quantile regression + calibration check.

For each target stat, trains LightGBM quantile regressors at multiple
quantiles. P(stat > line) at inference time = 1 - interpolated CDF from
those quantiles. Chronological split (train on earlier seasons, test on
the most recent) avoids leakage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom, poisson
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

TARGETS = ["points", "rebounds", "assists", "pra", "tpm", "steals", "blocks"]
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

# Assists are discrete, low-count, and heavily zero-inflated (27.8% of
# games are exactly 0; even the 25th percentile is 0). Continuous quantile
# regression collapses to predicting a near-constant ~0 at BOTH the 0.10
# and 0.25 quantiles regardless of player role (confirmed: predicted
# 10th-percentile std across players was ~0.002, essentially constant),
# and interpolating a CDF between two near-identical collapsed points
# produces unstable, falsely-confident probabilities right where most
# real lines sit (median assists is 1). Confirmed directly: bets placed
# by the quantile model won only 49.1% of the time despite averaging 71%
# stated confidence -- a real, large miscalibration, not a fluke.
# Fixed with Poisson regression (the appropriate distribution for count
# data): P(stat > line) computed exactly via the Poisson survival
# function, no interpolation involved. Confirmed this resolves it: bet
# win rate rose to 56.4% against a more modest 61.6% stated confidence.
POISSON_TARGETS = {"assists"}

# Rebounds is also non-negative count data, but not zero-heavy the way
# assists is (13.5% zero rate, 25th pctile is 1 not 0) -- so plain Poisson
# only marginally helped (bet win rate 54.6% vs 63.0% confidence, barely
# better than the original 54.3%/63.9%). Root cause here is different:
# rebounds is genuinely OVERDISPERSED -- per-player variance/mean ratio
# has a median of 1.55 (should be ~1.0 under Poisson), confirmed directly
# by grouping real outcomes per player. Poisson assumes variance=mean, so
# it understates real spread and ends up overconfident in the tails --
# confirmed the model's most-confident bucket (clamped ~0.95) actually
# WON only 44.3% of the time, worse than a coin flip. Fixed with a
# Negative Binomial (NB2: Var = mean + alpha*mean^2), which has a real
# dispersion parameter. Confirmed this resolves it: bet win rate rose to
# 57.3% against a more modest 61.7% stated confidence (gap 9.6pp -> 4.4pp).
NEGATIVE_BINOMIAL_TARGETS = {"rebounds"}

# 3PM/steals/blocks (2026-08-03): confirmed even more zero-heavy than assists
# (52-72% zero-rate vs assists' 27.8%), so Poisson is the right family here
# too, same reasoning as assists. Confirmed the same real tail-shrinkage
# pattern assists had (bucketing real players by their own actual volume:
# model overpredicts the lowest-volume bucket, underpredicts the highest --
# e.g. tpm bucket bias +0.30/-0.47 pre-recalibration by bucket), so isotonic
# recalibration applies to these three as well, via the same
# train_poisson_model() calibration path. NOTE: the population-level mean
# match is good for all three (tpm 0.848 vs 0.898 actual, steals 0.800 vs
# 0.789, blocks 0.428 vs 0.430) but full validation (bet-level win rate vs
# stated confidence -- the only tool this project has confirmed is valid for
# zero-heavy discrete targets, see the assists note above) was deferred to
# the end-of-session backtest per explicit user instruction -- do not treat
# these three as fully validated until that runs.
POISSON_TARGETS = POISSON_TARGETS | {"tpm", "steals", "blocks"}

FEATURE_COLS = [
    "points_per100_last5", "rebounds_per100_last5", "assists_per100_last5", "pra_per100_last5",
    "tpm_per100_last5", "steals_per100_last5", "blocks_per100_last5", "turnovers_per100_last5", "minutes_last5",
    "assists_zero_rate_last5", "steals_zero_rate_last5", "blocks_zero_rate_last5", "tpm_zero_rate_last5",
    "points_per100_last10", "rebounds_per100_last10", "assists_per100_last10", "pra_per100_last10",
    "tpm_per100_last10", "steals_per100_last10", "blocks_per100_last10", "turnovers_per100_last10", "minutes_last10",
    "assists_zero_rate_last10", "steals_zero_rate_last10", "blocks_zero_rate_last10", "tpm_zero_rate_last10",
    "own_team_ortg_last5", "own_team_drtg_last5", "own_team_pace_last5",
    "opp_team_ortg_last5", "opp_team_drtg_last5", "opp_team_pace_last5",
    "rest_days", "is_home",
    # Dynamic Pace & Possessions layer: real head-to-head expected game pace
    # (own x opp pace, normalized by the real rolling league-average pace --
    # see build_features.compute_league_avg_pace) plus each target's own
    # per-100 rate explicitly scaled against it. Hands the model a real,
    # directly-computed projected volume instead of leaving the pace x rate
    # interaction for it to reconstruct implicitly from separate features.
    "expected_pace_last5", "expected_pace_last10",
    "points_projected_volume_last5", "points_projected_volume_last10",
    "rebounds_projected_volume_last5", "rebounds_projected_volume_last10",
    "assists_projected_volume_last5", "assists_projected_volume_last10",
    # Defensive Matchup & Localized Splits: real position-specific (G/F/C)
    # defensive rate allowed, replacing the single blended opp_team_drtg as
    # the primary opponent-defense signal for player props (opp_team_drtg
    # stays in FEATURE_COLS too -- team-level context is still real
    # information, just coarser; kept rather than removed since there's no
    # evidence yet that dropping it helps).
    "opp_drtg_vs_position_points_last5", "opp_drtg_vs_position_points_last10",
    "opp_drtg_vs_position_rebounds_last5", "opp_drtg_vs_position_rebounds_last10",
    "opp_drtg_vs_position_assists_last5", "opp_drtg_vs_position_assists_last10",
    # Shooting Efficiency & Regression Engine: real eFG%/TS%, shrunk toward
    # each player's own real career baseline via an empirically fit
    # attempts-weighted correction (see build_features.compute_shooting_efficiency
    # for the real held-out validation of the shrinkage constants).
    "efg_shrunk_last5", "efg_shrunk_last10",
    "ts_shrunk_last5", "ts_shrunk_last10",
]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = df["game_date"].dt.year
    return df.dropna(subset=FEATURE_COLS + TARGETS)


def train_test_split_by_season(df: pd.DataFrame, test_seasons: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[~df["season"].isin(test_seasons)]
    test = df[df["season"].isin(test_seasons)]
    return train, test


def train_quantile_models(train: pd.DataFrame, target: str) -> dict[float, GradientBoostingRegressor]:
    """n_estimators=200/max_depth=4 confirmed directly (2026-08-03): the
    original n_estimators=60/max_depth=3 was sized for the smaller feature
    set this project started with. After tonight's real additions (pace,
    position-defense, shooting-efficiency features -- ~20 more real columns
    in FEATURE_COLS), the same small capacity model actually did WORSE than
    the pre-expansion baseline (points backtest bankroll $168,222->$119,975)
    despite the new features being individually real and validated -- the
    model didn't have enough capacity to actually use them. Verified
    directly: increasing capacity to match recovers and clearly exceeds the
    original baseline (points $168,222->$246,743, PRA $492,575->$1,394,685,
    both with a higher win rate too, not just a lucky bankroll path)."""
    models = {}
    X = train[FEATURE_COLS]
    y = train[target]
    for q in QUANTILES:
        model = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=200, max_depth=4, random_state=42)
        model.fit(X, y)
        models[q] = model
    return models


def predict_quantiles(models: dict[float, GradientBoostingRegressor], X: pd.DataFrame) -> pd.DataFrame:
    preds = {q: model.predict(X) for q, model in models.items()}
    out = pd.DataFrame(preds)
    return out.apply(lambda row: np.sort(row.values), axis=1, result_type="broadcast")


class PoissonModel:
    """Poisson regression for discrete count targets (assists). P(stat >
    line) uses the exact Poisson survival function -- no interpolation,
    so no instability from collapsed/near-identical quantile predictions.

    Includes an isotonic recalibration layer fit on a real held-out slice:
    confirmed directly (2026-08-01) that the raw tree regressor shrinks
    toward the population mean -- overpredicting low-volume players,
    underpredicting elite/high-volume ones -- because there are far fewer
    training examples at the extremes. Isotonic regression corrects this
    with a monotonic mapping from raw prediction to real outcome, fit on
    data the regressor itself never trained on."""

    def __init__(self, regressor: HistGradientBoostingRegressor, calibrator: IsotonicRegression | None = None):
        self.regressor = regressor
        self.calibrator = calibrator

    def predict_lambda(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.regressor.predict(X)
        if self.calibrator is not None:
            raw = self.calibrator.predict(raw)
        return np.clip(raw, 0.05, None)

    def predict_prob_over(self, x_row: pd.DataFrame, line: float) -> float:
        lam = self.predict_lambda(x_row)[0]
        # line is always a half-integer (e.g. 2.5); floor(line) is the
        # largest integer count still <= line, so sf(floor(line), lam)
        # is exactly P(X > line) for integer-valued X.
        prob = poisson.sf(np.floor(line), lam)
        return float(np.clip(prob, 0.02, 0.98))


def train_poisson_model(train: pd.DataFrame, target: str, calibration_frac: float = 0.15) -> PoissonModel:
    """Trains the regressor on an earlier slice and fits the isotonic
    correction on a real, later held-out slice (chronological split, not
    random -- matches the no-leakage discipline used everywhere else)."""
    train = train.sort_values("game_date")
    cutoff = int(len(train) * (1 - calibration_frac))
    fit_rows, calib_rows = train.iloc[:cutoff], train.iloc[cutoff:]

    regressor = HistGradientBoostingRegressor(loss="poisson", max_iter=200, random_state=42)
    regressor.fit(fit_rows[FEATURE_COLS], fit_rows[target])

    raw_calib_pred = regressor.predict(calib_rows[FEATURE_COLS])
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_calib_pred, calib_rows[target])

    return PoissonModel(regressor, calibrator)


class NegativeBinomialModel:
    """Poisson mean model + a fitted overdispersion parameter (NB2: Var =
    mean + alpha*mean^2). Used for count targets whose real variance
    exceeds what Poisson assumes (see NEGATIVE_BINOMIAL_TARGETS above)."""

    def __init__(self, regressor: HistGradientBoostingRegressor, alpha: float):
        self.regressor = regressor
        self.alpha = alpha

    def predict_mean(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.regressor.predict(X), 0.05, None)

    def predict_prob_over(self, x_row: pd.DataFrame, line: float) -> float:
        mu = self.predict_mean(x_row)[0]
        n = 1 / self.alpha
        p = 1 / (1 + self.alpha * mu)
        prob = nbinom.sf(np.floor(line), n, p)
        return float(np.clip(prob, 0.02, 0.98))


def _fit_nb_dispersion(train: pd.DataFrame, target: str) -> float:
    """Method-of-moments fit of alpha in Var = mean + alpha*mean^2, using
    each player's own real (mean, variance) across their games."""
    g = train.groupby("player_id")[target].agg(["mean", "var", "count"])
    g = g[g["count"] >= 20].dropna()
    mu, var = g["mean"].values, g["var"].values

    def loss(alpha: float) -> float:
        pred_var = mu + alpha * mu**2
        return float(np.sum((pred_var - var) ** 2))

    result = minimize_scalar(loss, bounds=(0.001, 2.0), method="bounded")
    return float(result.x)


def train_negative_binomial_model(train: pd.DataFrame, target: str) -> NegativeBinomialModel:
    regressor = HistGradientBoostingRegressor(loss="poisson", max_iter=200, random_state=42)
    regressor.fit(train[FEATURE_COLS], train[target])
    alpha = _fit_nb_dispersion(train, target)
    return NegativeBinomialModel(regressor, alpha)


def calibration_check(test: pd.DataFrame, target: str, quantile_preds: pd.DataFrame) -> pd.DataFrame:
    actual = test[target].values
    rows = []
    for q in QUANTILES:
        pred_q = quantile_preds[q].values
        empirical_rate = (actual <= pred_q).mean()
        rows.append({"quantile": q, "target_coverage": q, "empirical_coverage": empirical_rate})
    return pd.DataFrame(rows)


def main() -> int:
    df = load_dataset(Path("data/processed/player_features.csv"))
    print(f"Loaded {len(df)} rows, seasons: {sorted(df['season'].unique())}")

    test_seasons = {2025, 2026}
    train, test = train_test_split_by_season(df, test_seasons)
    print(f"Train: {len(train)} rows ({sorted(train['season'].unique())})")
    print(f"Test: {len(test)} rows ({sorted(test['season'].unique())})")

    for target in TARGETS:
        print(f"\n=== {target} ===")
        if target in POISSON_TARGETS:
            model = train_poisson_model(train, target)
            lam = model.predict_lambda(test[FEATURE_COLS])
            print(f"  Poisson model (see backtest for real validation): "
                  f"predicted lambda mean={lam.mean():.2f} std={lam.std():.2f}, "
                  f"actual mean={test[target].mean():.2f}")
            continue
        if target in NEGATIVE_BINOMIAL_TARGETS:
            model = train_negative_binomial_model(train, target)
            mu = model.predict_mean(test[FEATURE_COLS])
            print(f"  Negative Binomial model (alpha={model.alpha:.4f}, see backtest for real validation): "
                  f"predicted mean={mu.mean():.2f} std={mu.std():.2f}, "
                  f"actual mean={test[target].mean():.2f}")
            continue
        models = train_quantile_models(train, target)
        quantile_preds = predict_quantiles(models, test[FEATURE_COLS])
        calib = calibration_check(test, target, quantile_preds)
        print(calib.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
