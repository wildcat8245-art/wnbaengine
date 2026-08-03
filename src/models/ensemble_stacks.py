"""Wraps the validated ensemble stacks (src/backtest/backtest_points_ensemble_stack.py,
src/backtest/backtest_count_ensemble_stack.py) behind the SAME interface
predict_props.py already expects from a single model, so the live pipeline
(predict_prob_over, predict_point_estimate, parametric_simulation_prob_over,
the trend-conflict guard, etc.) needs no changes beyond swapping which
training function it calls. See SESSION_NOTES.md for the per-target
validated backtest numbers before this was wired in.

Per-player predictions are cached (keyed by player name) because the live
loop calls predict_prob_over / predict_point_estimate / parametric_
simulation_prob_over separately for the same (player, target) across
multiple bookmakers -- without caching, the RandomForest's per-tree
prediction loop would redo the same expensive work multiple times per pick.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor, QuantileRegressor, Ridge

from src.backtest.backtest_count_ensemble_stack import prob_over as count_prob_over
from src.backtest.backtest_count_ensemble_stack import stack_predict_rate, train_rate_stack
from src.backtest.backtest_points_ensemble_stack import (
    QUANTILES,
    prob_over_from_quantiles,
    stack_predict_quantiles,
    train_stack,
)
from src.models.train_baseline_model import NEGATIVE_BINOMIAL_TARGETS


class _QuantileLeaf:
    """A single quantile level's blended prediction, exposing .predict()
    so the wrapper below can support model[q].predict(x_row) exactly like
    the raw {quantile: fitted_model} dict the rest of the codebase expects."""

    def __init__(self, stack: "StackedQuantileModel", q: float):
        self.stack = stack
        self.q = q

    def predict(self, x_row: pd.DataFrame) -> np.ndarray:
        blended = self.stack._blended(x_row)
        idx = QUANTILES.index(self.q)
        return np.array([blended[idx]])


class StackedQuantileModel:
    """Real 3-base-learner stack (GBM + RandomForest + linear QuantileRegressor,
    quantile-consistent meta-learner) for a quantile-family target (points, pra).
    Duck-types as a {quantile: fitted_model} dict AND exposes predict_prob_over,
    matching both call sites in predict_props.py."""

    def __init__(self, gbm_models, rf, lin_models, meta_models):
        self.gbm_models = gbm_models
        self.rf = rf
        self.lin_models = lin_models
        self.meta_models = meta_models
        self._cache: dict[str, np.ndarray] = {}

    def _blended(self, x_row: pd.DataFrame) -> np.ndarray:
        key = str(x_row.index[0])
        if key not in self._cache:
            preds = stack_predict_quantiles(self.gbm_models, self.rf, self.lin_models, self.meta_models, x_row)
            self._cache[key] = preds[0]
        return self._cache[key]

    def keys(self):
        return QUANTILES

    def values(self):
        return [_QuantileLeaf(self, q) for q in QUANTILES]

    def __getitem__(self, q: float) -> _QuantileLeaf:
        return _QuantileLeaf(self, q)

    def predict_prob_over(self, x_row: pd.DataFrame, line: float) -> float:
        return prob_over_from_quantiles(self._blended(x_row), line)


def train_stacked_quantile_model(train: pd.DataFrame, target: str) -> StackedQuantileModel:
    gbm_models, rf, lin_models, meta_models = train_stack(train, target)
    return StackedQuantileModel(gbm_models, rf, lin_models, meta_models)


class _RateCacheMixin:
    """Shared caching so predict_prob_over / predict_point_estimate /
    parametric_simulation_prob_over (called separately, same player, per
    bookmaker row) don't repeat the RandomForest's per-tree prediction
    loop for the same player more than once."""

    def _rate(self, x_row: pd.DataFrame) -> np.ndarray:
        key = str(x_row.index[0])
        if key not in self._cache:
            self._cache[key] = stack_predict_rate(self.gbm, self.rf, self.glm, self.meta, x_row)
        return self._cache[key]


class StackedPoissonRateModel(_RateCacheMixin):
    """Rate ensemble for a Poisson-family target (assists/tpm/steals/blocks).
    Exposes ONLY predict_lambda (no predict_mean/alpha) -- matching exactly
    what PoissonModel exposes, so predict_point_estimate's and
    parametric_simulation_prob_over's `hasattr(model, "predict_lambda")`
    checks route here and not into the NegBinom branch."""

    def __init__(self, gbm, rf, glm, meta, target: str):
        self.gbm, self.rf, self.glm, self.meta = gbm, rf, glm, meta
        self.target = target
        self._cache: dict[str, np.ndarray] = {}

    def predict_lambda(self, x_row: pd.DataFrame) -> np.ndarray:
        return self._rate(x_row)

    def predict_prob_over(self, x_row: pd.DataFrame, line: float) -> float:
        rate = self._rate(x_row)[0]
        return count_prob_over(rate, line, self.target, None)


class StackedNegBinomRateModel(_RateCacheMixin):
    """Rate ensemble for the NegBinom target (rebounds). Exposes ONLY
    predict_mean + alpha (no predict_lambda) -- matching exactly what
    NegativeBinomialModel exposes, so the hasattr checks route here
    correctly instead of taking the Poisson branch."""

    def __init__(self, gbm, rf, glm, meta, alpha: float, target: str):
        self.gbm, self.rf, self.glm, self.meta = gbm, rf, glm, meta
        self.alpha = alpha
        self.target = target
        self._cache: dict[str, np.ndarray] = {}

    def predict_mean(self, x_row: pd.DataFrame) -> np.ndarray:
        return self._rate(x_row)

    def predict_prob_over(self, x_row: pd.DataFrame, line: float) -> float:
        rate = self._rate(x_row)[0]
        return count_prob_over(rate, line, self.target, self.alpha)


def train_stacked_rate_model(train: pd.DataFrame, target: str):
    gbm, rf, glm, meta, alpha = train_rate_stack(train, target)
    if target in NEGATIVE_BINOMIAL_TARGETS:
        return StackedNegBinomRateModel(gbm, rf, glm, meta, alpha, target)
    return StackedPoissonRateModel(gbm, rf, glm, meta, target)
