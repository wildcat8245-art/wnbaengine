"""Real backtest of the garbage-time/blowout-risk minutes adjustment.

Not covered by backtest_props.py -- that script's walk-forward simulation
never calls predict_props.py's live blowout-adjustment logic. This
specifically tests: for real historical starters in real 2025-2026 games
where we also have a real historical spread (fetch_historical_spreads.py),
does APPLYING the adjustment (scaling minutes_last5/10 by the fitted ratio
before prediction) produce a real prediction closer to the real actual
outcome than NOT applying it -- only where the adjustment is meaningful
(ratio < MIN_MEANINGFUL_ADJUSTMENT, i.e. real, visible blowout risk).

Uses the same train (<=2024) / test (2025-2026) split as every other model
in this project -- no leakage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.garbage_time import fit_blowout_minutes_curve, minutes_adjustment_ratio, MIN_MEANINGFUL_ADJUSTMENT
from src.models.train_baseline_model import (
    FEATURE_COLS, POISSON_TARGETS, NEGATIVE_BINOMIAL_TARGETS,
    load_dataset, train_test_split_by_season, train_poisson_model,
    train_negative_binomial_model, train_quantile_models,
)
from src.models.predict_props import predict_point_estimate

TARGETS_TO_CHECK = ["points", "rebounds", "assists", "tpm"]


def main() -> int:
    df = load_dataset(Path("data/processed/player_features.csv"))
    train, test = train_test_split_by_season(df, test_seasons={2025, 2026})

    spreads = pd.read_csv("data/raw/historical_spreads.csv", dtype={"game_id": str})
    spreads["abs_spread"] = spreads["home_spread_median"].abs()
    test = test.copy()
    test["game_id"] = test["game_id"].astype(str)
    test = test.merge(spreads[["game_id", "abs_spread"]], on="game_id", how="inner")
    test = test[test["starter"] == True]  # noqa: E712
    print(f"Real 2025-2026 starter rows with a real historical spread: {len(test)}")

    blowout_model = fit_blowout_minutes_curve(
        Path("data/raw/historical_spreads.csv"), Path("data/raw/player_boxscores_historical.csv")
    )
    test["ratio"] = test["abs_spread"].apply(lambda s: minutes_adjustment_ratio(blowout_model, s))
    affected = test[test["ratio"] < MIN_MEANINGFUL_ADJUSTMENT]
    print(f"Rows where the real adjustment is meaningful (ratio < {MIN_MEANINGFUL_ADJUSTMENT}): {len(affected)}\n")

    if affected.empty:
        print("No real rows meet the meaningful-adjustment threshold in the 2025-2026 test set -- "
              "cannot validate the adjustment's real effect on accuracy from this sample.")
        return 0

    for target in TARGETS_TO_CHECK:
        if target in POISSON_TARGETS:
            model = train_poisson_model(train, target)
        elif target in NEGATIVE_BINOMIAL_TARGETS:
            model = train_negative_binomial_model(train, target)
        else:
            model = train_quantile_models(train, target)

        errors_unadjusted, errors_adjusted = [], []
        for _, row in affected.iterrows():
            x_row = row[FEATURE_COLS].to_frame().T.copy()
            actual = row[target]

            proj_raw, _ = predict_point_estimate(model, x_row)
            errors_unadjusted.append(abs(proj_raw - actual))

            x_adj = x_row.copy()
            x_adj["minutes_last5"] = x_adj["minutes_last5"] * row["ratio"]
            x_adj["minutes_last10"] = x_adj["minutes_last10"] * row["ratio"]
            proj_adj, _ = predict_point_estimate(model, x_adj)
            errors_adjusted.append(abs(proj_adj - actual))

        mae_unadjusted = float(np.mean(errors_unadjusted))
        mae_adjusted = float(np.mean(errors_adjusted))
        better = "ADJUSTED wins (real improvement)" if mae_adjusted < mae_unadjusted else "UNADJUSTED wins (adjustment hurts here)"
        print(f"{target}: unadjusted MAE={mae_unadjusted:.3f}, adjusted MAE={mae_adjusted:.3f} -- {better}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
