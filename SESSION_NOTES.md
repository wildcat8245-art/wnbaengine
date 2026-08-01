# WNBA Player Prop System — Session State

_Last updated: 2026-07-31. Read this first if picking the project back up._

## 1. What this is

A from-scratch WNBA player-prop prediction and betting-decision system, built after
deliberately erasing an earlier, much larger WNBA project (`files_extracted`, deleted
2026-07-31 per explicit user request) and a small pre-existing stub in this same repo
(`get_odds.py`, kept). The design is adapted from a real academic paper the user
supplied (Montrucchio, Barbierato & Gatti, *"Uncertainty-Aware Machine Learning for
NBA Forecasting in Digital Betting Markets"*, Information 2026) — calibrated
probabilistic forecasting + fractional-Kelly/EV-threshold decision layer + a
falsification test to prove any backtest profit isn't an artifact. The paper's own
methods (LSTM + Monte Carlo Dropout, built for NBA-scale data) were deliberately
**not** copied wholesale — see §4 for what was adapted and why.

## 2. Architecture (what actually exists, and where)

```
src/data/
  wnba_client.py              # RapidAPI wnba-api: schedule + box score fetch
  fetch_historical_boxscores.py  # bulk multi-season historical puller (resumable)
  build_game_season_types.py  # schedule-only walk -> game_id -> season type + real US date
  odds_client.py               # the-odds-api.com: live player prop odds client
  fetch_daily_props.py         # daily snapshot of today's real prop odds (quota-safe)
src/features/
  build_features.py           # rolling windows, per-100, opponent adjustment, rest/home
src/models/
  train_baseline_model.py     # quantile / Poisson / Negative Binomial models + calibration check
  predict_props.py            # live board: real odds + model -> Kelly-sized picks
src/backtest/
  backtest_props.py           # walk-forward backtest + falsification test
data/raw/                     # gitignored — regenerate by rerunning the fetch scripts
  player_boxscores_historical.csv   # 63,672 rows, 2,759 games, 563 players, 2015-2026
  game_season_types.csv             # 3,017 games -> season type + corrected US date
  daily_props/props_YYYY-MM-DD.csv  # one snapshot per day fetch_daily_props.py was run
data/processed/
  player_features.csv         # gitignored — output of build_features.py, 52,932 rows
get_odds.py                    # OLD, pre-existing direct-DraftKings-scrape script.
                                # Now confirmed BLOCKED (403, DK added bot protection).
                                # Not used by anything below. Candidate for removal.
```

**Env vars (Windows User scope, persisted across sessions):**
- `RAPIDAPI_KEY` — wnba-api.p.rapidapi.com. 14,000/period limit; this session's heavy
  historical backfilling pushed it into negative "remaining" on the quota header at
  one point, but requests still succeeded (HTTP 200) when checked — not actually
  blocking as of 2026-07-31, but don't assume infinite headroom.
- `ODDS_API_KEY` — the-odds-api.com. 500/month limit, showed 500/500 remaining as of
  2026-07-31 (fresh cycle). User's own words: "once we use up all the request I'm
  going to get the paid version." Budget deliberately — each event's player-prop pull
  costs ~3 units (one per market), so a full day's slate (~5 games) costs ~15 units.
- `SPORTSBOOK_RAPIDAPI_KEY` — leftover from the deleted old project. Unused by
  anything in this codebase. Not removed, just inert.

## 3. Pipeline, end to end

1. `python -m src.data.fetch_historical_boxscores --start-year 2015 --end-year 2026`
   — full historical pull. Resumable (skips game_ids already in the CSV).
2. `python -m src.data.build_game_season_types` — needed once (or after any schedule
   gap) to get season-type + corrected US game dates. See §4 for why this exists.
3. `python -m src.features.build_features` — builds `data/processed/player_features.csv`.
4. `python -m src.backtest.backtest_props` — walk-forward backtest + falsification
   test, train ≤2024 / test 2025-2026.
5. `python -m src.data.fetch_daily_props` — today's real live prop odds snapshot
   (FanDuel/DraftKings/BetRivers/BetOnline.ag; NOT PrizePicks — that's a pick'em/
   multiplier product incompatible with standard decimal-odds Kelly staking, and
   wasn't rebuilt in this new system).
6. `python -m src.models.predict_props` — trains final models on ALL available
   history, reads the latest `daily_props/` snapshot, prints a real Kelly-staked board.

## 4. Real bugs found and fixed this session (read before touching the data pipeline)

1. **Schedule month-only queries silently return a partial window, not the full
   month.** `wnbaschedule?year=Y&month=M` (no `day`) returns some fixed ~9-day
   window, confirmed directly. Fixed by iterating every day of the month explicitly
   in `iter_completed_game_ids`.
2. **Game dates were wrong** — originally read from the response's outer dict key
   (the query anchor day, not the real game day) and even after fixing that, still
   wrong because the API stamps games in UTC and a US-evening game can land on the
   next UTC day. Fixed with `wnba_client.iso_utc_to_us_game_date()` (converts to
   US/Eastern before taking the date). This was caught by noticing one player had 8
   rows all dated the same day — impossible for a real season.
3. **A single bad day aborted an entire month's collection.** `iter_completed_game_ids`
   had no per-day try/except; one 500 error (2019-07-05, reproducible) silently
   dropped ~75 games for that whole month. Fixed to catch per-day failures and
   continue — recovered the 2019-07 gap on retry (66/78 games back).
4. **Preseason games were polluting the dataset** (~128 games) — real WNBA doubleheaders
   in early May, confirmed via each game's own `season.slug` field
   (`"preseason"` vs `"regular-season"`/`"post-season"`). Filtered out.
5. **Backtest synthetic-line bias caused a $21-trillion bankroll** on the first run —
   the synthetic line (used because no real historical prop lines exist anywhere)
   was constructed as `floor(rolling_avg) - 0.5` / `floor(rolling_avg) + 0.5`, both
   systematically below the true expected value, guaranteeing the model looked
   "right" on almost every bet. Fixed with unbiased `round(x*2)/2` rounding.
6. **Assists: real, severe miscalibration** — bet-level win rate was 49.1% despite
   71% average stated confidence (bankroll backtest collapsed $1,000 → $0.63). Root
   cause: assists are zero-inflated (27.8% of games are exactly 0; even the 25th
   percentile is 0). Continuous quantile regression (tried both
   `GradientBoostingRegressor` and `HistGradientBoostingRegressor`) collapsed to
   predicting a near-constant ~0 regardless of player role — confirmed by checking
   the predicted quantile's std across players (~0.002, essentially constant) and by
   testing more capacity (no effect at all) and a hurdle model (classifier + positive-
   only quantile regression — the classifier itself was well-calibrated, verified, but
   the recombination math had a bug and made pooled coverage worse; abandoned, not
   shipped). **Real fix: Poisson regression** (`HistGradientBoostingRegressor(loss=
   "poisson")` for the mean, exact `scipy.stats.poisson.sf` for P(over line), no
   interpolation). Confirmed: bet win rate → 56.4% vs 61.6% stated; backtest bankroll
   → tens of thousands (healthy order of magnitude, not the collapse).
   **Important side-finding**: the pooled/marginal quantile-coverage table is the
   WRONG validation tool for a discrete zero-heavy variable — a row whose true P(0)
   exceeds the target quantile level correctly gets predicted 0, but its own achieved
   coverage is its own P(0), which inflates the pooled average regardless of model
   quality. Validate this class of model economically (bet-level win rate vs stated
   confidence), not via the marginal quantile table.
7. **Rebounds: a milder version of the same symptom, different cause.** Bet-level
   gap was 9.6pp (54.3% actual vs 63.9% stated), and the most-confident bucket
   (~95%) actually LOST — 44.3% win rate, worse than random. Rebounds isn't
   zero-heavy (13.5% zero rate) so the quantile model's predictions did vary
   properly by player (std 1.1–2.9 across quantiles, not collapsed) — plain Poisson
   only marginally helped (8.4pp gap). Real cause: **overdispersion** — real
   per-player variance/mean ratio has a median of 1.55 (should be ~1.0 under
   Poisson), confirmed directly from real per-player (mean, variance) pairs. Fixed
   with a **Negative Binomial** model (NB2: Var = mean + α·mean², α fit via
   method-of-moments from real data = 0.0942). Confirmed: bet win rate → 57.3% vs
   61.7% stated (gap 9.6pp → 4.4pp); backtest bankroll rebounds → ~$202K (from ~$25K).

## 5. Deliberately NOT built / explicitly rejected

- **Monte Carlo simulation** (empirical-covariance game resampling, the old deleted
  project's core engine) — rejected up front as unsuitable for WNBA's much smaller
  per-player game-log sample than NBA.
- **RNN + Monte Carlo Dropout** (the paper's actual uncertainty technique) — started
  as a planned comparison model, installed PyTorch, hit a missing Visual C++
  Redistributable DLL dependency mid-install, then the user explicitly said to
  **drop this entirely, permanently** ("forget about the rnn mc dropout... period").
  The draft file was deleted. Do not re-raise this unless the user asks again.
  (PyTorch CPU wheel may still be `pip`-installed but is unused; the VC++
  redistributable install was killed mid-run and its final state was never checked
  — irrelevant unless this is revisited.)
- **PrizePicks integration** — not rebuilt. the-odds-api.com covers FanDuel,
  DraftKings, BetRivers, BetOnline.ag (real, licensed, confirmed working), which is
  actually a better foundation for Kelly staking than PrizePicks' multiplier format
  anyway.

## 6. Known open items for next session

- **No real historical prop lines exist anywhere.** Every backtest number uses a
  synthetic line (player's own trailing average). Absolute backtest bankroll
  figures ($100K–$1M+ range) are **not real profitability estimates** — they show
  the model beats a naive baseline, nothing more. Same caveat the source paper
  makes about its own player-prop synthetic-odds test.
- Points (4.5pp) and PRA (2.1pp) still carry a small, unaddressed overconfidence gap
  — much smaller than assists/rebounds were, not investigated further, "good enough"
  per the pattern established this session.
- `get_odds.py` (old direct DraftKings scrape) is confirmed broken (403) and unused
  — candidate for deletion, not yet removed. Its Claude Code skill
  (`.claude/skills/draftkings-odds/SKILL.md`) points at it and is equally stale.
- No unit tests anywhere in this codebase.
- `predict_props.py` reads whatever is the *most recent* file in `data/raw/daily_props/`
  — you must run `fetch_daily_props.py` fresh the same day before trusting the board.
- A calibration chart was published as a Claude Artifact this session
  (backtest snapshot, not live-updating) — see auto-memory for the URL if needed;
  it reflects one specific backtest run and will drift if models are retrained.
