"""Fractional-Kelly staking, split out of predict_props.py so it can be
imported by both predict_props.py and the ensemble-stack backtest modules
without a circular import (ensemble_stacks.py needs the stack backtest
modules, which need kelly_stake, which used to live in predict_props.py,
which needs ensemble_stacks.py)."""

from __future__ import annotations

KELLY_FRACTION = 0.3
MAX_STAKE_FRACTION = 0.05


def kelly_stake(prob: float, decimal_odds: float, bankroll: float) -> float:
    b = decimal_odds - 1
    q = 1 - prob
    f = (b * prob - q) / b
    f = max(f, 0.0) * KELLY_FRACTION
    f = min(f, MAX_STAKE_FRACTION)
    return f * bankroll
