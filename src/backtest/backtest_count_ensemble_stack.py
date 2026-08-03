"""Real diverse-algorithm ensemble stack for the 5 count-family targets
(rebounds=NegBinom, assists/tpm/steals/blocks=Poisson) -- companion to
backtest_points_ensemble_stack.py, which handles the 2 quantile-family
targets (points/pra). NOT wired into predict_props.py yet.

Different design from the points/pra stack, on purpose: these targets are
modeled as a single predicted RATE (mean count), not a set of quantiles, so
the right ensemble blends rate estimates (a squared-error problem) rather
than quantile levels (a pinball-loss problem). Concretely:

Base learners (genuine diversity -- boosting, bagging, and linear, matching
the same principle used for the points/pra stack):
  1. HistGradientBoostingRegressor(loss="poisson") -- same family as the
     current production model, tree boosting
  2. RandomForestRegressor -- bagging, squared-error loss
  3. PoissonRegressor -- linear GLM with a log link and Poisson deviance
     loss; genuinely different functional form (linear, not tree-based)
     AND appropriately matches these targets' real distribution, unlike
     using a plain linear-regression base learner would

Meta-learner: Ridge regression (NOT QuantileRegressor -- that was the right
fix for the points/pra stack specifically because it blends quantile
levels; here we're blending predicted MEANS, and squared-error loss is the
textbook-correct choice for that, not a mismatch the way plain
LinearRegression was for a quantile blend).

Overdispersion (rebounds only): reuses the existing method-of-moments
alpha fit (`_fit_nb_dispersion`) directly -- it's fit from real per-player
(mean, variance) data, not from any particular regressor's predictions, so
it doesn't need to change for a blended mean.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor, Ridge

from src.models.staking import kelly_stake
from src.models.train_baseline_model import (
    FEATURE_COLS,
    NEGATIVE_BINOMIAL_TARGETS,
    POISSON_TARGETS,
    _fit_nb_dispersion,
    load_dataset,
    train_test_split_by_season,
)

warnings.filterwarnings("ignore", category=UserWarning)

VIG_DECIMAL_ODDS = 1.909
EV_THRESHOLD = 1.02
STARTING_BANKROLL = 1000.0
COUNT_TARGETS = ["rebounds", "assists", "tpm", "steals", "blocks"]


def train_rate_stack(train: pd.DataFrame, target: str):
    train_sorted = train.sort_values("game_date")
    cutoff = int(len(train_sorted) * 0.85)
    fit_rows, holdout_rows = train_sorted.iloc[:cutoff], train_sorted.iloc[cutoff:]
    X_fit, y_fit = fit_rows[FEATURE_COLS], fit_rows[target]
    X_hold = holdout_rows[FEATURE_COLS]

    gbm = HistGradientBoostingRegressor(loss="poisson", max_iter=200, random_state=42)
    gbm.fit(X_fit, y_fit)

    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_fit, y_fit)

    glm = PoissonRegressor(alpha=0.01, max_iter=500)
    glm.fit(X_fit, y_fit)

    base_hold = np.column_stack([
        np.clip(gbm.predict(X_hold), 0.01, None),
        np.clip(rf.predict(X_hold), 0.01, None),
        np.clip(glm.predict(X_hold), 0.01, None),
    ])
    meta = Ridge(alpha=1.0, positive=True)
    meta.fit(base_hold, holdout_rows[target])

    alpha = _fit_nb_dispersion(train, target) if target in NEGATIVE_BINOMIAL_TARGETS else None
    return gbm, rf, glm, meta, alpha


def stack_predict_rate(gbm, rf, glm, meta, X: pd.DataFrame) -> np.ndarray:
    base = np.column_stack([
        np.clip(gbm.predict(X), 0.01, None),
        np.clip(rf.predict(X), 0.01, None),
        np.clip(glm.predict(X), 0.01, None),
    ])
    return np.clip(meta.predict(base), 0.05, None)


def prob_over(rate: float, line: float, target: str, alpha: float | None) -> float:
    if target in NEGATIVE_BINOMIAL_TARGETS:
        n = 1 / alpha
        p = 1 / (1 + alpha * rate)
        prob = nbinom.sf(np.floor(line), n, p)
    else:
        prob = poisson.sf(np.floor(line), rate)
    return float(np.clip(prob, 0.02, 0.98))


def run_target(train: pd.DataFrame, test: pd.DataFrame, target: str) -> None:
    print(f"\n=== {target} ===", flush=True)
    raw_estimate = test[f"{target}_per100_last5"] / 100 * test["possessions"]
    test = test.copy()
    test["synthetic_line"] = np.round(raw_estimate * 2) / 2

    print(f"  training base learners (HistGBM-poisson, RandomForest, PoissonRegressor GLM)...", flush=True)
    gbm, rf, glm, meta, alpha = train_rate_stack(train, target)
    print(f"  meta (Ridge) coefs (gbm, rf, glm) = {meta.coef_}, intercept={meta.intercept_:.3f}", flush=True)

    test["stack_rate"] = stack_predict_rate(gbm, rf, glm, meta, test[FEATURE_COLS])
    test = test.sort_values("game_date").reset_index(drop=True)

    bankroll = STARTING_BANKROLL
    bets = []
    for _, row in test.iterrows():
        line = row["synthetic_line"]
        actual = row[target]
        prob = prob_over(row["stack_rate"], line, target, alpha)
        ev = prob * VIG_DECIMAL_ODDS
        stake = kelly_stake(prob, VIG_DECIMAL_ODDS, bankroll) if ev > EV_THRESHOLD else 0.0
        if stake > 0:
            won = actual > line
            bankroll += stake * (VIG_DECIMAL_ODDS - 1) if won else -stake
            bets.append({"prob": prob, "won": won})

    bets_df = pd.DataFrame(bets)
    print(f"  Stack bankroll: {bankroll:.2f} ({len(bets_df)} bets)", flush=True)
    if len(bets_df):
        print(f"  Real win rate: {bets_df['won'].mean():.1%}, stated avg confidence: {bets_df['prob'].mean():.1%}", flush=True)
        bets_df["bucket"] = pd.cut(bets_df["prob"], bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        print(bets_df.groupby("bucket", observed=True).agg(n=("won", "size"), win_rate=("won", "mean")), flush=True)


def main() -> int:
    df = load_dataset(Path("data/processed/player_features.csv"))
    train, test = train_test_split_by_season(df, test_seasons={2025, 2026})
    for target in COUNT_TARGETS:
        run_target(train, test, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
