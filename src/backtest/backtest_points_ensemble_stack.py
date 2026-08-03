"""Real diverse-algorithm ensemble stack for the points target, validated
against the single capacity-fixed quantile model baseline ($246,742.58,
see backtest_props.py). NOT wired into predict_props.py yet -- this is a
standalone validation script, kept here so it can be rerun without
retraining from scratch in a scratchpad file that disappears at session end.

Base learners (genuine diversity, not three flavors of the same algorithm --
see SESSION_NOTES.md for why a prior GBM-vs-HistGBM diversity check found
0.9947 correlation and got rejected):
  1. GradientBoostingRegressor -- sequential tree boosting
  2. RandomForestRegressor     -- bagging (quantile via per-tree prediction
     distribution, a standard technique -- each tree in the forest gives one
     point prediction per row; the empirical quantile of that per-row
     distribution across trees approximates the model's predictive quantile)
  3. QuantileRegressor         -- linear, genuinely different functional form

Meta-learner: QuantileRegressor per quantile level (NOT plain LinearRegression
-- that was tried first and bankrupted the backtest to $0.00, because squared-
error loss fits the conditional MEAN regardless of which quantile it's
supposedly blending, collapsing the q=0.1..0.9 spread and producing
overconfident, badly miscalibrated probabilities). Matching the meta-learner's
loss to the quantile it represents is what fixed it.

Real backtest result (2026-08-03): $503,568.95 (1895 bets) vs the single-model
baseline's $246,742.58 (2008 bets) -- roughly 2x. Sanity-checked, not just
taken at face value (the capacity bug earlier this session taught not to
trust an unusually large improvement without checking):
  - Real win rate 57.6% vs stated avg confidence 60.2% -- close, honest.
  - Bucket calibration is good through 50-80% stated confidence.
  - The >80%-stated-confidence bucket is real but tiny (n=19) and actually
    LOSES (50%/44% win rate) -- same small-sample-tail-overconfidence pattern
    already seen with rebounds earlier this session. Don't trust any single
    high-confidence pick from this stack until more data exists there.
  - Bankroll grows smoothly through the bet sequence (not one lucky bet):
    $5.5K at 25% of bets, $35K at 50%, $164K at 75%, $504K at 100%.
  - The 2x (not just a few points) is plausible specifically because Kelly
    staking compounds a real edge exponentially over ~1900 sequential bets --
    a modest real calibration improvement in the tails can produce a large
    final gap without needing a giant single-bet improvement.

Scope: points only. The other 6 targets (rebounds/assists/pra/tpm/steals/
blocks) already use Poisson/NegBinom models with their own real fixes
(isotonic recalibration, NB2 overdispersion fit) -- this stack has NOT been
built or validated for them. Doing so means repeating this same (fairly
slow -- the linear QuantileRegressor step uses an LP solver) process per
target, a real time/compute cost, not yet authorized.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import QuantileRegressor

from src.models.predict_props import kelly_stake
from src.models.train_baseline_model import FEATURE_COLS, load_dataset, train_test_split_by_season

warnings.filterwarnings("ignore", category=UserWarning)

VIG_DECIMAL_ODDS = 1.909
EV_THRESHOLD = 1.02
STARTING_BANKROLL = 1000.0
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
TARGET = "points"


def rf_quantile_predict(rf: RandomForestRegressor, X: pd.DataFrame, q: float) -> np.ndarray:
    """Empirical quantile of the forest's per-tree point predictions -- a
    standard way to get a quantile estimate out of a model that isn't
    natively a quantile regressor."""
    all_preds = np.array([tree.predict(X) for tree in rf.estimators_])
    return np.quantile(all_preds, q, axis=0)


def train_stack(train: pd.DataFrame):
    """Returns (gbm_models, rf, lin_models, meta_models) all fit on a real
    chronological fit/holdout split within train -- same discipline used
    elsewhere in this codebase for isotonic calibration."""
    train_sorted = train.sort_values("game_date")
    cutoff = int(len(train_sorted) * 0.85)
    fit_rows, holdout_rows = train_sorted.iloc[:cutoff], train_sorted.iloc[cutoff:]
    X_fit, y_fit = fit_rows[FEATURE_COLS], fit_rows[TARGET]
    X_hold = holdout_rows[FEATURE_COLS]

    gbm_models = {}
    for q in QUANTILES:
        m = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=200, max_depth=4, random_state=42)
        m.fit(X_fit, y_fit)
        gbm_models[q] = m

    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_fit, y_fit)

    lin_models = {}
    for q in QUANTILES:
        m = QuantileRegressor(quantile=q, alpha=0.01, solver="highs")
        m.fit(X_fit, y_fit)
        lin_models[q] = m

    meta_models = {}
    for q in QUANTILES:
        base_preds = np.column_stack([
            gbm_models[q].predict(X_hold),
            rf_quantile_predict(rf, X_hold, q),
            lin_models[q].predict(X_hold),
        ])
        meta = QuantileRegressor(quantile=q, alpha=0.01, solver="highs")
        meta.fit(base_preds, holdout_rows[TARGET])
        meta_models[q] = meta

    return gbm_models, rf, lin_models, meta_models


def stack_predict_quantiles(gbm_models, rf, lin_models, meta_models, X: pd.DataFrame) -> np.ndarray:
    preds = {}
    for q in QUANTILES:
        base_preds = np.column_stack([
            gbm_models[q].predict(X),
            rf_quantile_predict(rf, X, q),
            lin_models[q].predict(X),
        ])
        preds[q] = meta_models[q].predict(base_preds)
    pred_df = pd.DataFrame(preds)
    return np.sort(pred_df.values, axis=1)


def prob_over_from_quantiles(preds: np.ndarray, line: float) -> float:
    preds = np.maximum.accumulate(preds)
    if line <= preds[0]:
        prob = 1 - QUANTILES[0] / 2
    elif line >= preds[-1]:
        prob = (1 - QUANTILES[-1]) / 2
    else:
        prob = 1 - np.interp(line, preds, QUANTILES)
    return float(np.clip(prob, 0.05, 0.95))


def main() -> int:
    df = load_dataset(Path("data/processed/player_features.csv"))
    train, test = train_test_split_by_season(df, test_seasons={2025, 2026})
    test = test.reset_index(drop=True)

    raw_estimate = test[f"{TARGET}_per100_last5"] / 100 * test["possessions"]
    test[f"{TARGET}_synthetic_line"] = np.round(raw_estimate * 2) / 2

    print("Training stack (GBM + RandomForest + linear QuantileRegressor, quantile-consistent meta-learner)...")
    gbm_models, rf, lin_models, meta_models = train_stack(train)

    print("Generating blended predictions on the real test set...")
    stack_preds = stack_predict_quantiles(gbm_models, rf, lin_models, meta_models, test[FEATURE_COLS])
    for i, q in enumerate(QUANTILES):
        test[f"stack_q{q}"] = stack_preds[:, i]

    test = test.sort_values("game_date").reset_index(drop=True)

    bankroll = STARTING_BANKROLL
    bets = []
    for _, row in test.iterrows():
        line = row[f"{TARGET}_synthetic_line"]
        actual = row[TARGET]
        preds = np.array([row[f"stack_q{q}"] for q in QUANTILES])
        prob = prob_over_from_quantiles(preds, line)
        ev = prob * VIG_DECIMAL_ODDS
        stake = kelly_stake(prob, VIG_DECIMAL_ODDS, bankroll) if ev > EV_THRESHOLD else 0.0
        if stake > 0:
            won = actual > line
            bankroll += stake * (VIG_DECIMAL_ODDS - 1) if won else -stake
            bets.append({"prob": prob, "won": won})

    bets_df = pd.DataFrame(bets)
    print(f"\nStack bankroll: {bankroll:.2f} ({len(bets_df)} bets)")
    print(f"Real win rate: {bets_df['won'].mean():.1%}, stated avg confidence: {bets_df['prob'].mean():.1%}")
    bets_df["bucket"] = pd.cut(bets_df["prob"], bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    print(bets_df.groupby("bucket", observed=True).agg(n=("won", "size"), win_rate=("won", "mean")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
