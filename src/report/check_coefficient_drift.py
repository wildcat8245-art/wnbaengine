"""Real-Time Execution & Feedback Logging: coefficient-drift check.

Re-fits this project's empirically-determined constants (NB2 overdispersion
alphas, eFG%/TS% shrinkage strengths) against the latest real data and
reports how far they've drifted from what's currently hardcoded in
train_baseline_model.py / build_features.py.

Deliberately PROPOSAL-ONLY -- prints a real, human-readable comparison and
writes nothing back to any source file. Auto-applying a refit constant to
live decision logic without a human checking it first would violate this
project's own CLAUDE.md rule 1 (backtest before live) and rule 6 (ask
before executing changes to real-money decision logic). Review the numbers
below and decide by hand whether an update is warranted.

Usage:
    python -m src.report.check_coefficient_drift
"""

from __future__ import annotations

from pathlib import Path

from src.features.build_features import load_and_clean, EFG_SHRINKAGE_K, TS_SHRINKAGE_K
from src.models.train_baseline_model import (
    load_dataset, NEGATIVE_BINOMIAL_TARGETS, _fit_nb_dispersion,
)


def check_nb2_alphas() -> None:
    df = load_dataset(Path("data/processed/player_features.csv"))
    print(f"=== NB2 overdispersion alpha ({'/'.join(sorted(NEGATIVE_BINOMIAL_TARGETS))}) ===")
    for target in sorted(NEGATIVE_BINOMIAL_TARGETS):
        alpha = _fit_nb_dispersion(df, target)
        print(f"  {target}: current real fit alpha = {alpha:.4f}")
    print()


def check_shooting_shrinkage() -> None:
    print("=== eFG%/TS% shrinkage strength (K) ===")
    raw_path = Path("data/raw/player_boxscores_historical.csv")
    df = load_and_clean(raw_path)
    df["season"] = df["game_date"].dt.year
    df = df.sort_values(["player_id", "game_date"])
    grp = df.groupby("player_id")

    for col in ["fgm", "fga", "tpm", "fta", "points"]:
        df[f"{col}_sum_last5"] = grp[col].transform(lambda s: s.shift(1).rolling(5, min_periods=5).sum())
        df[f"{col}_sum_career"] = grp[col].transform(lambda s: s.shift(1).expanding(min_periods=10).sum())

    train = df[df["season"] <= 2024]

    def fit_and_report(name: str, current_k: float, is_efg: bool) -> None:
        career_fga = train["fga_sum_career"]
        career_tsa = career_fga + 0.44 * train["fta_sum_career"]
        if is_efg:
            baseline = (train["fgm_sum_career"] + 0.5 * train["tpm_sum_career"]) / career_fga
        else:
            baseline = train["points_sum_career"] / (2 * career_tsa)

        fga5 = train["fga_sum_last5"]
        tsa5 = fga5 + 0.44 * train["fta_sum_last5"]
        actual_this_game = (
            (train["fgm"] + 0.5 * train["tpm"]) / train["fga"] if is_efg
            else train["points"] / (2 * (train["fga"] + 0.44 * train["fta"]))
        )
        mask = fga5.notna() & baseline.notna() & train["fga"].gt(0) & (career_fga > 0)

        best_k, best_mse = None, float("inf")
        for k in [100, 300, 500, 750, 1000, 1500, 2000]:
            if is_efg:
                made_equiv = train["fgm_sum_last5"] + 0.5 * train["tpm_sum_last5"]
                pred = (made_equiv + k * baseline) / (fga5 + k)
            else:
                pred = (train["points_sum_last5"] + 2 * k * baseline) / (2 * (tsa5 + k))
            mse = float(((pred[mask] - actual_this_game[mask]) ** 2).mean())
            if mse < best_mse:
                best_mse, best_k = mse, k

        print(f"  {name}: currently hardcoded K={current_k:.0f}, fresh refit best K={best_k} "
              f"({'same order of magnitude' if abs(best_k - current_k) / current_k < 0.5 else 'MEANINGFULLY DIFFERENT -- review'})")

    fit_and_report("eFG%", EFG_SHRINKAGE_K, is_efg=True)
    fit_and_report("TS%", TS_SHRINKAGE_K, is_efg=False)
    print()


def main() -> int:
    print("Coefficient drift check -- PROPOSAL ONLY, nothing auto-applied.\n")
    check_nb2_alphas()
    check_shooting_shrinkage()
    print("Review the above by hand. Update the hardcoded constants in "
          "train_baseline_model.py / build_features.py yourself if warranted -- "
          "this script never writes to source files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
