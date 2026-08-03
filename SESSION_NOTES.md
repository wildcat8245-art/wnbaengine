# WNBA Player Prop System — Session State

_Last updated: 2026-08-02 (midday session). Read this first if picking the
project back up. §9 below is new since the 2026-08-01 save; §7 and §8 are new
since the 2026-07-31 save._

## 1. What this is

A from-scratch WNBA player-prop prediction and betting-decision system, built after
deliberately erasing an earlier, much larger WNBA project (`files_extracted`, deleted
2026-07-31 per explicit user request) and a small pre-existing stub in this same repo
(`get_odds.py`, kept). Calibrated probabilistic forecasting + fractional-Kelly/
EV-threshold decision layer + a falsification test to prove any backtest profit
isn't an artifact — see §4 for what was built and why.

## 2. Architecture (what actually exists, and where)

```
src/data/
  wnba_client.py              # RapidAPI wnba-api: schedule + box score fetch
  fetch_historical_boxscores.py  # bulk multi-season historical puller (resumable)
  build_game_season_types.py  # schedule-only walk -> game_id -> season type + real US date
  odds_client.py               # the-odds-api.com: live player prop odds client
  fetch_daily_props.py         # daily snapshot of today's real prop odds (quota-safe)
  injuries_client.py           # wnba-api /injuries: live injury reports + usage-vacuum detection
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

- **CORRECTION (2026-08-01, later session):** this section previously claimed
  "Monte Carlo simulation... rejected up front as unsuitable for WNBA" as if the
  user had agreed to that. **That was false and was never the user's decision** —
  unlike the RNN+MC-Dropout rejection below, it carries no user quote, and the
  user has since stated directly that Monte Carlo simulation was the explicit
  purpose of this project. A real Monte Carlo layer (20,000-draw simulation
  sampling each pick's own fitted Poisson/NegBinom/quantile distribution,
  cross-checked against the closed-form probability) was added 2026-08-01 in
  `src/models/predict_props.py` (`monte_carlo_prob_over`). Do not re-assert the
  old "rejected" framing. The exact form of Monte Carlo the user wants long-term
  (parametric distribution sampling, as built, vs. empirical/historical game-log
  resampling, closer to the old deleted project's approach) was still being
  clarified as of this note — check for a newer decision before assuming either.
- **RNN + Monte Carlo Dropout** — started
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
  the model beats a naive baseline, nothing more.
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

## 7. 2026-08-01 session: real live board run, more real bugs found and fixed

Pulled real props for 2026-08-01's actual games (LVA@CHI, NY@PHX — 350 real prop
rows, `data/raw/daily_props/props_2026-08-01.csv`) and ran the live board for real.
This surfaced two more real defects that hadn't shown up in the backtest:

1. **`predict_props.py` was silently discarding every Under.** It only ever
   evaluated the "Over" side of each line. Fixed: now computes `prob_over` once per
   (player, market, line), then evaluates BOTH sides against their own real quoted
   odds (Over and Under have different odds due to vig) and reports whichever side
   actually has the edge. This roughly quadrupled the number of real recommended
   bets (21 → 82) because a lot of real value was sitting on the Under side the
   whole time and was never being shown.

2. **A real, systematic shrinkage bias in the assists Poisson model, found via a
   live sanity check, not the backtest.** Comparing the model's predicted lambda
   against players' simple recent form for real star players (Alyssa Thomas, A'ja
   Wilson, Chelsea Gray, etc.) showed the model consistently underpredicting elite/
   high-volume players and overpredicting bench players — classic tree-model
   shrinkage toward the population mean (far fewer training examples at the
   extremes). Confirmed across all 456 players by volume bucket, not just a few
   cherry-picked names.

   **Two real self-corrections happened while investigating this, both worth
   remembering as methodology lessons:**
   - First diagnostic compared model predictions to a *naive last-5-game average*
     as if it were ground truth. That overstates the real problem — some of that
     gap is the model legitimately using more information (last-10, opponent,
     rest, home) than a naive last-5 average, not bias. **Always validate against
     real held-out outcomes, not a naive proxy.**
   - Second diagnostic (applied to points/PRA) compared the model's predicted
     *median* (0.5 quantile) against the *mean* of real outcomes per bucket. For
     any right-skewed stat, median < mean by definition even for a perfectly
     calibrated model — that gap was a comparison artifact, not a real defect.
     Points/PRA's real bias (median-vs-actual-median) turned out to be small and
     non-monotonic — the earlier "points off by -1.65" alarm was wrong.
     **Always compare like-for-like: median predictions against actual medians,
     mean predictions against actual means.**

   **Real, validated fix for assists**: isotonic regression recalibration
   (`sklearn.isotonic.IsotonicRegression`), fit on a genuine held-out slice
   *within* the training data (chronologically later 15% of the ≤2024 training
   rows — NOT the 2025-2026 test set, to avoid circularity). `train_poisson_model()`
   in `train_baseline_model.py` now does this split internally and returns a
   `PoissonModel` with an optional `calibrator`. Verified on the true, fully
   held-out 2025-2026 test set: per-bucket bias dropped from as much as -0.45/+0.36
   down to under 0.09 in every bucket, no more systematic pattern. This is the
   only model with isotonic recalibration applied — rebounds/points/PRA were
   checked properly (see above) and didn't show the same severe problem, so they
   were left as-is.

3. **The board never showed *why*** — only player/line/odds/probability/EV, no
   supporting numbers. Every pick required manually digging into a player's recent
   games to understand the reasoning. Fixed: every row now includes
   `stat_per100_last5`, `stat_per100_last10`, and `minutes_last5_vs_last10`
   directly, so the supporting context is visible by default, not something that
   has to be extracted pick-by-pick after the fact.

4. **New: a trend-conflict "stay away" guard** (`trend_conflict_flag()` in
   `predict_props.py`). Found live: Rebecca Allen's model recommendation was Under
   on both points and rebounds, but her real recent form was clearly trending UP
   (last-5 per-100 rate and minutes both well above last-10) — the model and her
   own visible trend actively disagreed, and nothing caught that before the user
   found it by manually pulling her game log. The guard fires when a player's
   per-100 rate AND minutes both move ≥25%/≥15% in the direction opposite the
   model's recommended side, and forces that pick's stake to 0 regardless of EV.
   Verified live: caught Rebecca Allen (both props) and Natasha Mack (points,
   trending down while model said Over) on the 2026-08-01 board — 9 duplicate
   book-rows pulled, recommended-bet count went 79 → 73.

## 8. Explicitly rejected again this session (don't re-raise)

- **A target of "70-80% accuracy."** The user asked for this explicitly; it was
  refused directly and honestly. No legitimate sports-prop system hits that —
  real edges in this space run around 55-58% against fair odds, and that's
  already the difference between profit and loss at real sportsbook odds. If asked again, explain
  this rather than promise a number.
- This session was very long (referenced by the user as roughly 4pm to past
  midnight) and included real user frustration after two consecutive comparison
  mistakes (see §7) produced alarming-but-wrong bias numbers before being
  corrected. The concrete, lasting responses were: (a) fix the real bug that
  existed (isotonic recalibration for assists), (b) add supporting context to
  every board row by default, (c) add the trend-conflict guard. If a future
  session inherits distrust from this one, the fastest way to rebuild it is
  pointing at these three concrete, git-committed, verified changes rather than
  re-asserting that things work.

## 9. 2026-08-02 session: empirical Monte Carlo engine, live injury/usage-vacuum checks

Commit `105e9474` (see `CLAUDE.md` for the process rules that came out of this
same session). Real changes made, in order of how they change what the board
actually recommends:

1. **`empirical_resample_prob_over()` in `predict_props.py`** — a second,
   independent probability engine: bootstrap-resamples `n_sims` draws (with
   replacement) directly from a player's own last 20 real game outcomes, no
   parametric shape assumed. `EMPIRICAL_MC_TARGETS` decides, per market, which
   engine actually drives the live pick — chosen by backtest, not by default.
   The 2026-08-02 backtest run (~8,900 held-out 2025-2026 bets) showed the
   empirical engine wins for points (264K vs 178K synthetic-line bankroll) but
   loses badly for rebounds (13K vs 385K), assists (7.8K vs 21K), and PRA (247K
   vs 1.24M) — so only points uses it live; the other three keep the
   parametric model (NB2 / isotonic-Poisson / quantile). Falls back to the
   parametric probability when a player has fewer than `MC_MIN_POOL_GAMES` (8)
   real prior games, flagged in the board output, never silent.
2. **`backtest_props.py` rewritten to walk three engines side by side** on
   identical bets: parametric, empirical_mc (the same function that drives the
   live board), and a falsification baseline (p=0.5 always) — so the routing
   decision in (1) is validated against real held-out outcomes before being
   trusted, not asserted.
3. **New `src/data/injuries_client.py`** — live injury report client (wnba-api
   `/injuries` endpoint, same RAPIDAPI_KEY as `wnba_client.py`). Wired into
   `predict_props.py`: a confirmed "Out" (or out-for-season) player has their
   pick's stake forced to 0 regardless of EV, board says so loudly. If the
   injury API call itself fails, the board prints a visible warning rather than
   silently presenting an unchecked board as checked.
4. **Usage-vacuum detection** (`build_usage_vacuum_map`) — cross-references
   each board player's own name against the literal text of every injury
   report's long/short comment. Real WNBA injury writeups routinely name the
   specific teammates expected to absorb missing minutes; this just checks
   whether a player's name is directly present in that text — no guessed
   minutes-redistribution model, no hardcoded name pairs. Confirmed directly
   (2026-08-02) it recovers real cases (Julie Allemand/Marina Mabrey, Janelle
   Salaun/Gabby Williams, Diamond Miller/Aaliyah Edwards, Isabelle
   Harrison/Aneesah Morrow) with zero manual entries. Under picks on the
   named beneficiary are auto-excluded (more usage argues against Under); Over
   picks get an informational note only, since the vacuum reinforces those.
5. **`fetch_daily_props.py`** — now filters fetched events to those whose real
   US-calendar game date (UTC `commence_time` converted to US/Eastern) matches
   today, before spending any quota on their prop odds. Confirmed directly
   that the events endpoint was returning games 1-2 days out, which had been
   silently saved into a file named after today's date and treated as today's
   slate.
6. **`wnba_client.iso_utc_to_us_game_date()`** now accepts both wnba-api's
   (ESPN-style, no seconds) and the-odds-api's (seconds included) timestamp
   formats, since this same function is now called from both clients.
7. **`train_baseline_model.py`** — `random_state=42` added to both
   `HistGradientBoostingRegressor` instances (Poisson and Negative Binomial)
   for reproducibility across reruns.

**Live board run confirms this is wired in, not just written**: the
2026-08-02 board (`predictions_2026-08-02_full_report.md`, 253 combinations
evaluated, 59 recommended after all checks) shows real examples of every
mechanism above firing on the same day's real slate — e.g. Brittney Griner
(confirmed Out, knee) auto-excluded despite EV 1.301, and multiple Janelle
Salaun / Ariel Atkins Under picks excluded by the usage-vacuum check with the
real sourced injury text quoted in the board's own excluded-picks section.

**Known gap this update closes**: this file previously did not document any
of the above even though the code and a live board run already reflected it
— commit `105e9474` touched `SESSION_NOTES.md` only to strip false
paper-citation framing (see §5), not to record its own feature work. Point
this out again if a future session finds SESSION_NOTES.md lagging behind
the actual code — check `git log` and the code itself, not just this file's
own claims (per `CLAUDE.md` rule 5).
