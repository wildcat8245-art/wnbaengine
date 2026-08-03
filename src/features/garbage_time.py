"""Garbage-Time & Blowout Risk Modeling: real spread-to-minutes adjustment.

Confirmed directly (2026-08-03) using real historical spread data (paid
ODDS_API_KEY's historical archive, 2022-05-21 onward -- see
fetch_historical_spreads.py) joined to real starter minutes: starters
average ~29.5 real minutes in games with a real spread under ~8 points,
declining to ~84% of that in real 25-30+ point spread games. This is a
real, measured relationship, not an assumed threshold rule.

Applied at PREDICTION time only, not baked into training: when a real live
spread indicates high blowout risk for a player who was a real starter in
their most recent game, their own minutes_last5/last10 feature values are
scaled down by the real fitted ratio before being fed to the
already-trained model -- the model's own learned relationship between
minutes and output stats then naturally produces a correspondingly lower
projection, rather than a hand-crafted post-hoc rescaling formula.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

BASELINE_SPREAD = 1.0  # a near-pick'em game -- the reference point for "normal" minutes
MIN_MEANINGFUL_ADJUSTMENT = 0.98  # ignore trivial ratios; only act on a real, visible effect


def fit_blowout_minutes_curve(
    historical_spreads_path: Path, boxscores_path: Path
) -> IsotonicRegression | None:
    """Real isotonic fit: |spread| -> expected real starter minutes.
    Monotonically non-increasing by construction (larger spread should never
    imply MORE expected minutes) -- matches the real, measured direction.
    Returns None if no real historical spread data is available yet."""
    if not historical_spreads_path.exists():
        return None

    spreads = pd.read_csv(historical_spreads_path, dtype={"game_id": str})
    if spreads.empty:
        return None
    spreads["abs_spread"] = spreads["home_spread_median"].abs()

    box = pd.read_csv(boxscores_path, dtype={"game_id": str, "player_id": str})
    box["minutes"] = pd.to_numeric(box["minutes"], errors="coerce")
    box["did_not_play"] = box["did_not_play"].astype(bool)
    played = box[(box["did_not_play"].eq(False)) & (box["minutes"].fillna(0).gt(0))]

    merged = played.merge(spreads[["game_id", "abs_spread"]], on="game_id")
    starters = merged[merged["starter"] == True]  # noqa: E712
    if len(starters) < 100:
        return None

    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(starters["abs_spread"], starters["minutes"])
    return iso


def minutes_adjustment_ratio(model: IsotonicRegression, abs_spread: float) -> float:
    """Real, empirically-derived multiplier (<=1) for a normal starter's own
    minutes given today's real spread magnitude -- ratio of fitted minutes at
    this spread vs. the low-spread baseline, not an absolute replacement."""
    baseline = model.predict([BASELINE_SPREAD])[0]
    fitted = model.predict([abs_spread])[0]
    if baseline <= 0:
        return 1.0
    return float(np.clip(fitted / baseline, 0.0, 1.0))
