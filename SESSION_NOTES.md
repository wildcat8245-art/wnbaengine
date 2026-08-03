# WNBA Player Prop System — Session State

_Last updated: 2026-08-03 (continued, later still). Read this first if
picking the project back up. **§13 is new and supersedes §12.1's routing
table — the quantile-regressor capacity fix (see §13.1) changed the points
engine comparison after §12.1 was written.** §12 documents real backtest
results for everything in §11, and TWO REVERTS (on/off splits, garbage-time)
that §11 does not reflect. §11 is long — read it in full before touching
predict_props.py, train_baseline_model.py, or build_features.py. §10 is new
since the 2026-08-02 save; §9 since 2026-08-01; §7/§8 since 2026-07-31._

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
  fetch_historical_spreads.py  # real historical spread pull (2022-05-21+), matches to our real games
  fetch_play_by_play.py        # real historical shot-event pull (zone classification, raw only)
  fetch_substitution_events.py # real substitution-event pull, recent-season sample
src/report/
  build_board.py               # renders the latest predictions_*.csv into templates/board_template.html
  grade_predictions.py         # joins saved predictions_*.csv boards to real box scores, grades win rate
  track_clv.py                 # real Closing Line Value vs. actual closing odds (needs paid historical access)
  check_coefficient_drift.py   # PROPOSAL-ONLY refit check for NB2 alpha / eFG-TS shrinkage K, never auto-applies
src/features/
  build_features.py           # rolling windows, per-100, opponent adjustment, rest/home, pace,
                                # position defense, shooting-efficiency shrinkage (see §11)
  on_off_splits.py             # real teammate on/off split calc -- NOT wired into predict_props.py (see §12, reverted)
  garbage_time.py              # real spread-to-minutes isotonic fit -- NOT wired into predict_props.py (see §12, disabled)
src/models/
  train_baseline_model.py     # quantile / Poisson / Negative Binomial models + calibration check
  predict_props.py            # live board: real odds + model -> Kelly-sized picks
src/backtest/
  backtest_props.py           # walk-forward backtest + falsification test (all 7 targets, see §12)
  backtest_garbage_time.py     # real backtest of the (now-disabled) blowout minutes adjustment
  backtest_on_off_splits.py    # real backtest of the (now-reverted) on/off split mechanism
data/raw/                     # gitignored — regenerate by rerunning the fetch scripts
  player_boxscores_historical.csv   # ~63,700 rows, 2,768 games, through 2026-08-02
  game_season_types.csv             # 3,017 games -> season type + corrected US date
  daily_props/props_YYYY-MM-DD.csv  # one snapshot per day fetch_daily_props.py was run
  daily_game_lines/game_lines_YYYY-MM-DD.csv  # real game-level total/spread snapshot
  historical_spreads.csv       # real 2022-05-21+ spreads, 1,008 games (fetch_historical_spreads.py)
  shot_events_historical.csv   # 376,094 real shot events, 2,762/2,768 games (fetch_play_by_play.py) -- RAW ONLY, unused downstream (see §11.6)
  substitution_events_recent.csv  # 27,867 real substitution events, 535 games 2025-2026 (fetch_substitution_events.py)
data/processed/
  player_features.csv         # gitignored — output of build_features.py
  predictions_YYYY-MM-DD.csv  # one saved snapshot per board run -- the record graded later
  board_YYYY-MM-DD.html       # gitignored — build_board.py output, publish as a Claude Artifact
  graded_predictions.csv      # gitignored — grade_predictions.py output, real win-rate history
  clv_report.csv              # gitignored — track_clv.py output, real Closing Line Value history
templates/
  board_template.html         # locked-in board visual design (see §10) -- don't redesign ad hoc
get_odds.py                    # OLD, pre-existing direct-DraftKings-scrape script.
                                # Now confirmed BLOCKED (403, DK added bot protection).
                                # Not used by anything below. Candidate for removal.
```

**Env vars (Windows User scope, persisted across sessions):**
- `RAPIDAPI_KEY` — wnba-api.p.rapidapi.com. 14,000/period limit; has been reading
  NEGATIVE remaining (confirmed again 2026-08-03: -1,572/14,000) since the original
  2026-07-31 historical backfill, but requests still succeed (confirmed directly,
  real 200s with real data) — a soft/unenforced limit so far, not a hard block, but
  don't assume that holds forever, especially after the ~2,768-game play-by-play
  pull this session (see §11).
- `ODDS_API_KEY` — the-odds-api.com. **Upgraded to a paid plan 2026-08-03** (the key
  value itself changed — see `C:\Users\User\Desktop\odds paid api key.txt`). Quota
  is now 20,000/period (was 500/month free tier). Real, load-bearing consequence:
  **historical odds are now accessible** — confirmed directly, real historical
  game-level (h2h/spreads/totals) AND player-prop odds both return real data via
  `/v4/historical/sports/basketball_wnba/...`. **Real, confirmed limit: the archive
  only goes back to 2022-05-21** — querying 2015/2020 both return empty with
  `next_timestamp` pointing at that same 2022-05-21 date. Cost: historical
  events-listing is cheap (~1 unit/call); historical odds/props cost ~10 units per
  market requested (~10x the live per-event rate) — a full real-line backtest across
  2022-2026 (~200+ games, one snapshot each) would cost roughly 6,000 units, well
  within budget. **Not yet used for anything** — this session confirmed the
  capability and stopped there; no historical-odds backtest has been built yet.
- `SPORTSBOOK_RAPIDAPI_KEY` — **NOT actually unused — this was wrong.** Previously
  documented (and treated by prior sessions) as an inert leftover from the deleted
  old project. Confirmed directly 2026-08-03 it is a real, working, currently-unused
  API: `sportsbook-api2.p.rapidapi.com` (found by probing candidate hosts/paths,
  landing on `/v0/competitions` → real WNBA competition key `OjR0-whdp-ZWM1` →
  `/v0/competitions/{key}/events` → real events → `/v0/events/{key}/markets`). Gives
  real MONEYLINE/POINT_SPREAD/POINT_TOTAL odds (no player props) across 8 real books
  (BetParx, BetRivers, BetMGM, Bovada, DraftKings, ESPN BET, FanDuel, Fanatics) — a
  different, partially-overlapping book set than the-odds-api's game-lines call.
  Confirmed NO historical access (date/season query params are silently ignored,
  same 8 current/upcoming events returned regardless; no historical/archive endpoint
  exists). Rate limit: 150 requests per ~6.2-hour window, 500,000 hard cap. **Not
  wired into anything yet** — confirmed real and working, not yet used to add a
  second real book-consensus source alongside the-odds-api's game lines.

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

## 10. 2026-08-03 session: board visualization, closed-loop grading, real calibration check

1. **Live board run for real 2026-08-03 games** (SEA@NY, LVA@ATL, PHX@CHI) —
   fetched fresh (569 real prop rows), 90 unique props evaluated, 55
   recommended after all checks.

2. **`templates/board_template.html` + `src/report/build_board.py`** — the
   user saw the board rendered as a Claude Artifact, liked the visual design
   (dark charcoal/gold trading-terminal look: top-picks rail with full
   rationale text, sortable/filterable full table, expandable per-row
   detail, both light/dark themes), and asked to lock that exact style in
   permanently rather than have it redesigned each time. `build_board.py`
   reads whatever `predictions_*.csv` + matching `daily_props/props_*.csv`
   are most recent, fills the template's placeholders, and writes a
   self-contained `data/processed/board_<date>.html` safe to publish as an
   Artifact with no further editing. Do not hand-build a new report design
   for this system without the user asking for a different look.

3. **`src/report/grade_predictions.py` — the closed loop that was missing.**
   Predictions were being saved (`predictions_*.csv`) and real box scores
   were being re-fetched, but nothing ever checked one against the other.
   This script joins every saved board to the player's real box score for
   that exact game date and checks whether the picked side actually cleared
   the line — real win rate vs. stated confidence, a real bankroll trace,
   a per-market breakdown, and a check on whether flagged stay-away picks
   were right to be excluded.
   - Had to backfill `player_boxscores_historical.csv` first — it was
     stale (last real game was 2026-07-30) — via
     `fetch_historical_boxscores.py --start-year 2026 --end-year 2026`
     (resumable, cheap on the wnba-api quota) to get real outcomes through
     2026-08-02.
   - **First real result** (2026-08-02 board — the only one gradable so
     far, since 2026-08-03's games hadn't been played yet at the time this
     was run): 58 recommended bets, real win rate **48.3%** vs **63.1%**
     average stated confidence — a 14.8pp gap. Real bankroll trace: $1,000
     → $920.04. Rebounds (47.1%) and assists (44.4%) worse than points
     (52.2%). The 11 flagged stay-away picks with a real outcome would have
     won only 36.4% — the trend-conflict/injury/usage-vacuum guards were
     directionally right to exclude them.
   - **Explicit caveat, stated to the user directly and worth repeating to
     avoid over-reacting**: n=58 from a single day is nowhere near enough to
     conclude the live system is miscalibrated — this is the same
     small-sample-comparison trap this project already learned to avoid in
     §7 (naive-proxy and median-vs-mean comparison mistakes). Do not treat
     this one-day gap as proof of a live calibration problem in a future
     session — pull the accumulated history from `graded_predictions.csv`
     across many boards before drawing any real conclusion.

4. **User raised a real, correct critique of the feature set**: no
   head-to-head-vs-this-specific-opponent signal, no Vegas game
   total/spread signal, no game-script/blowout-risk signal — only
   *team-level* opponent D-rating/pace (rolling, from real box scores),
   home/away, and rest days. Discussed and agreed: head-to-head is low
   real value here (most players face a given opponent only 2-4x/season —
   mostly noise, and the team-level opponent rating already captures most
   of the real signal more reliably than a tiny per-matchup sample would).
   Vegas game total/spread was agreed as the real, worthwhile gap — a
   genuine market signal for pace/blowout risk the system currently ignores
   entirely, even though the same the-odds-api key already used for player
   props also carries it.

5. **Vegas game total/spread added — as informational context only, not
   wired into the decision engine.** Confirmed directly: this key's plan has
   no historical-odds access (`GET /v4/historical/...` returns
   `HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN`, a 401, zero quota cost to
   check). That means there is no way to backfill real historical lines to
   train a model feature on, or to backtest a total/spread-based adjustment
   against real past outcomes — and per `CLAUDE.md` rule 1, nothing goes
   into `predict_props.py`'s actual probability/EV/staking logic without
   that. So this was built as transparent context instead:
   - `src/data/odds_client.py`: `get_game_lines()` — bulk
     `/sports/{sport}/odds/` call (spreads+totals for the whole day's slate
     in one request; confirmed 2 quota units total, not per-event, unlike
     the per-event player-prop endpoint) + `flatten_game_lines_to_rows()`.
   - `fetch_daily_props.py`: also saves
     `data/raw/daily_game_lines/game_lines_YYYY-MM-DD.csv` every run
     (same quota-floor guard as the props fetch).
   - `predict_props.py`: joins each player to their game's consensus
     total/home-spread (median across books) via the daily props file's own
     `event_id`, and appends a `Vegas: total X, {home_team} {spread}.` note
     to every rationale + a `vegas_note` column — informational only, does
     not affect `prob_over`, EV, or stake. Real example from the 2026-08-03
     board: Golden State (home) at -13.5 against Toronto, a real, visible
     blowout-risk signal the model has no other way to see.

## 11. 2026-08-03 (continued): full layer-by-layer rebuild from the user's own model spec, two new API discoveries, two real process failures

**Read this whole section before changing `predict_props.py`,
`train_baseline_model.py`, or `build_features.py` again — a lot changed.**

### 11.0 Context: a real trust breakdown, mid-session, and two new CLAUDE.md rules

After §11's work began as an AI-initiated "prepare to rebuild" plan (a new
game-level moneyline/spread/total subsystem, started without being asked),
the user stopped the session, correctly identified this as unrequested scope
expansion, and demanded deletion. The first deletion attempt was itself
interrupted (a tool-call rejection) and silently never completed — this was
only caught by accident later in the same session when the files showed up
again as untracked in `git status` (see 11.7). Two new binding rules were
added to `CLAUDE.md` as a direct result — **rule 7** (never expand into a new
unrequested subsystem, even as an implied "next step") and **rule 8** (data-
source/cost tradeoffs, like whether to pay for historical odds access, are
the user's decision, not something to quietly design around). Read both
before starting any new subsystem or hitting a real data-access wall.

After that reset, the user walked through their own hypothetical model
design layer by layer, in their own words, asking for an honest real/math/
fantasy assessment of each piece before anything got built. Everything in
11.1–11.6 below was built only after that assessment and an explicit
"build it" from the user — this is the process to keep using on future
layers, not a one-time exception.

### 11.1 Two new real API capabilities discovered (both real, both tested directly)

1. **`ODDS_API_KEY` upgraded to a paid plan** (key value itself changed —
   see `C:\Users\User\Desktop\odds paid api key.txt`). 20,000/period quota
   (was 500/month). **Real historical odds now work** — both game-level
   (h2h/spreads/totals) and player-prop odds, confirmed via
   `/v4/historical/sports/basketball_wnba/...`. **Real, confirmed limit:
   archive only goes back to 2022-05-21** (2015/2020 queries return empty
   with `next_timestamp` pointing at that same date). Cost: ~1 unit for a
   historical events list, ~10 units per market for historical odds/props
   (~10x the live per-event rate). **Not used for anything yet** — capability
   confirmed and stopped there this session, per the new rule 8 discipline
   (surfaced to the user, not unilaterally built into a feature).
2. **`SPORTSBOOK_RAPIDAPI_KEY` is real and working — previous notes calling
   it an inert leftover were wrong**, never actually re-verified until this
   session (a real instance of the "verify before reporting" rule mattering).
   It's `sportsbook-api2.p.rapidapi.com`: real WNBA competition key
   `OjR0-whdp-ZWM1` → real events → real MONEYLINE/POINT_SPREAD/POINT_TOTAL
   odds across 8 real books (BetParx, BetRivers, BetMGM, Bovada, DraftKings,
   ESPN BET, FanDuel, Fanatics) — no player props, confirmed no historical
   access (date/season params silently ignored). Rate limit 150 req/~6.2hrs,
   500,000 hard cap. **Not wired into anything yet** — a real second
   book-consensus source for game lines, sitting unused.

### 11.2 Dynamic Pace & Possessions (`build_features.py`)

Real head-to-head expected game pace: `expected_pace_last{5,10} =
own_team_pace * opp_team_pace / league_avg_pace` (`compute_league_avg_pace`
— a real, ROLLING league-wide average, not a fixed constant, since real
season-average pace/scoring has shifted era to era). Each target's own
per-100 rate is then explicitly scaled against this expected pace into a
real `{points,rebounds,assists,tpm}_projected_volume_last{5,10}` feature,
handing the model a direct projection instead of leaving the pace-rate
interaction for it to reconstruct implicitly. Verified: formula checked by
hand against a real row (85.5 × 87.3 / 82.9 = 90.04, exact match). Wired
into `FEATURE_COLS`. Calibration re-checked after adding — held.

### 11.3 Defensive Matchup & Localized Splits (`build_features.py`)

Real position-specific (G/F/C — confirmed clean, only 3 real values, no
hybrids) defensive rate allowed: `opp_drtg_vs_position_{points,rebounds,
assists}_last{5,10}`, replacing team-wide DRtg as the primary opponent
signal (team-wide DRtg is kept too, not removed — no evidence dropping it
helps). Full 5-man lineup-combination splits (the literal spec) were
deliberately NOT built — real WNBA rotations don't generate enough repeated
5-man combinations in a ~40-game season to support that granularity
reliably; would be reporting noise as precision. Verified real and sane:
guards allow far more points/assists per-100 than centers (50.9/15.2 vs
18.0/2.6), matches real basketball. Wired into `FEATURE_COLS`.

### 11.4 Usage & On-Court Lineup Dynamics — real teammate on/off splits (`src/features/on_off_splits.py`, new file)

Real, GAME-level on/off splits (did a specific teammate play at all in this
game, not true on-court-stint overlap within the game — that finer version
is real and buildable from play-by-play, see 11.6, but is a separate, much
larger undertaking, deliberately not done). `compute_on_off_split()`
computes a player's real per-36-minute rate in real games a named teammate
did vs didn't play, **only reports a number when both conditions have at
least `MIN_OUT_GAMES=5` real games** — otherwise returns
`insufficient_data=True` rather than fabricating a split from noise. Wired
into `predict_props.py`'s existing usage-vacuum section: upgrades the
previously purely-qualitative flag (injury text merely *names* a
beneficiary) with a real measured `pct_change` per stat when sample size
allows. Verified directly on real data both ways: correctly refused a
2-real-game sample (Julie Allemand/Marina Mabrey), and produced real,
sane numbers on a sufficient sample (Jacy Sheldon's real rebound rate +38.6%
in real Chicago Sky games without Azura Stevens on court). Live board run
confirmed no errors, though no real vacuum case triggered the new code path
that specific run (empty vacuum_map that day) — the direct test above is
what actually validates the logic.

### 11.5 Shooting Efficiency & Regression Engine (`build_features.py`)

Real eFG%/TS% computed from real made/attempted **sums** over the rolling
window (not a naive mean-of-game-percentages, which would wrongly weight a
2-attempt game equal to a 15-attempt game), regressed toward each player's
own real career-to-date baseline via **empirically fit** attempts-weighted
shrinkage — not a guessed constant. Method: grid-searched shrinkage strength
K against real same-game shooting outcomes on train seasons (≤2024), then
verified the winning K's improvement holds on the true held-out 2025-2026
test set (same no-leakage discipline as the isotonic fits). Real, verified
results: **eFG% K=1000, test MSE 0.0915→0.0789 (raw last5 → shrunk); TS%
K=750, test MSE 0.0818→0.0708** — both genuine, out-of-sample error
reductions (~13-14%), not train-set-only improvements. "Contest level"
(defender proximity) was explicitly identified as fantasy and NOT built —
no data source we have access to carries defender-tracking data. New
columns: `efg_shrunk_last{5,10}`, `ts_shrunk_last{5,10}`, wired into
`FEATURE_COLS`.

### 11.6 Shot-zone breakdown — real historical play-by-play pull (`src/data/fetch_play_by_play.py`, new file)

Confirmed real play-by-play exists per game (`/wnbasummary`'s `plays`
array) with real shot outcomes, and — critically — a real shot distance
in the play's own free-text description (e.g. "Diana Taurasi makes 4-foot
two point shot"). Resumable fetcher (same pattern as
`fetch_historical_boxscores.py`) pulled the full historical set.

**Two real parsing bugs found and fixed during smoke-testing, before
trusting the output — both would have silently corrupted the data:**
1. `pointsAttempted` looked like it would distinguish a 2pt vs 3pt attempt,
   but it reads **0 for every missed shot regardless of shot value**
   (confirmed directly) — only meaningful on makes, where it just duplicates
   `scoreValue`. Fixed by using the play's own text label instead — both
   makes AND misses reliably say "three point" explicitly in real game
   text (confirmed: 29/29 sampled missed threes carried the label).
2. Layups/dunks routinely have **no distance figure in the text at all**
   ("misses layup", "makes dunk") — a distance-only zone rule dumped nearly
   all real rim attempts into "unknown" instead of "rim", and the measured
   rim make-rate came out at 21.4% on the test sample — LOWER than
   mid-range, backwards from real basketball. Fixed by checking the play's
   own TYPE text for layup/dunk/tip keywords first, before falling back to
   distance. After the fix, real basketball-sane numbers: rim 58.3% made
   (highest, correct), three 34.4%, mid-range zones 37.8%/42.1%.

**Final pull result: 376,094 real shot events across 2,762/2,768 historical
games** (6 games missed to transient read-timeouts — the fetcher is
resumable, rerun `python -m src.data.fetch_play_by_play` to pick those up,
it will skip everything already fetched). Output:
`data/raw/shot_events_historical.csv` (gitignored, `*.csv`).

**This is raw data only. Nothing downstream has been built yet** — no
rolling zone-rate features, no zone-based expected-eFG% model, nothing
wired into `build_features.py` or any trained model. That's the real next
step on this specific layer, not done this session.

### 11.7 Player Prop Projection Engine — 3PM/steals/blocks added as real modeled targets (`train_baseline_model.py`, `predict_props.py`, `odds_client.py`)

Points/Rebounds/Assists/PRA already satisfied this layer's actual spec
("independent micro-projections... free from sportsbook juice") — that's
the system's existing core design, not new work. The real gap was 3PM,
steals, and blocks having raw data (parsed since day one) but never being
fit as prediction targets at all.

Checked real distribution characteristics before picking a model family
(same discipline as assists/rebounds), **not assumed**: all three are more
zero-heavy than assists (tpm 59.9%, steals 52.4%, blocks 72.2% zero-rate,
vs assists' 27.8%), with mild-to-moderate overdispersion (variance/mean
ratio 1.07–1.23, milder than rebounds' 1.55). Poisson picked as the base
family (matches the zero-heavy precedent). **Confirmed the exact same real
tail-shrinkage bias assists had**, bucketing real players by their own
actual test-period volume (model overpredicts the lowest-volume bucket,
underpredicts the highest — e.g. tpm bucket bias +0.30/−0.47) — isotonic
recalibration applied via the existing `train_poisson_model()` path, same
mechanism as assists. **Real, important caveat, not resolved this
session**: the standard marginal-quantile/bucket check is documented in
this project as **the wrong validation tool** for zero-heavy discrete
targets (see §4 item 6) — real population-level means match well for all
three (tpm 0.848 vs 0.898 actual, steals 0.800 vs 0.789, blocks 0.428 vs
0.430) but **the only validation this project has confirmed is actually
meaningful for this class of model is the bet-level backtest, which has
NOT been run for these three yet** (deferred to end-of-session per
explicit user instruction — see 11.8). Do not treat tpm/steals/blocks as
fully validated until that happens.

**Real market-availability check, not assumed**: `player_threes` (3PM) is
a genuine live market — confirmed 5 real bookmakers on today's real slate.
**`player_steals` and `player_blocks` have ZERO bookmaker coverage across
all 4 real games checked** — no sportsbook the-odds-api tracks currently
offers these markets for WNBA. So: 3PM is wired all the way into
`odds_client.PROP_MARKETS`, `predict_props.MARKET_TO_TARGET`, and is live
on the real board (confirmed: e.g. Jonquel Jones tpm Under 1.5 @ 2.04,
EV 1.377, real pick on the 2026-08-03 board). Steals/blocks are real,
trained, isotonic-recalibrated models with **no live market to attach to**
— same situation `pra` has been in since the start.

### 11.8 Outstanding / next session

- **No comprehensive bet-level backtest has been run across all of
  11.2–11.7's changes together.** Individual lightweight sanity checks
  were done as each piece was built (population-mean matches, calibration
  re-checks, direct function tests) — deliberately NOT the full
  `backtest_props.py` bankroll/win-rate run, per explicit user instruction
  to defer that to the end. **Run it before trusting any of tonight's
  changes for a real bet** — per `CLAUDE.md` rule 1, this is not optional.
- Shot-zone breakdown (11.6): raw data collected, nothing built on top of
  it yet. Real next step: rolling per-player zone-attempt-rate features,
  and a zone-mix-weighted expected-eFG% (a player who takes more of their
  shots at the rim should have a higher expected efficiency baseline than
  raw season eFG% alone implies).
- `SPORTSBOOK_RAPIDAPI_KEY`'s real game-line data (11.1) is not yet
  combined with the-odds-api's game lines into a single stronger
  real-book-consensus signal — currently only the-odds-api's game lines
  feed `predict_props.py`'s Vegas context.
- The paid `ODDS_API_KEY`'s real historical-odds access (11.1) has not
  been used for anything — the natural next step is a real backtest of
  the player-prop models against real 2022-2026 historical lines instead
  of the synthetic trailing-average line, which would finally answer
  whether this system actually beats real historical vig, not just a
  naive baseline.
- Six historical games (see 11.6) failed the play-by-play pull on
  transient timeouts — rerun the fetcher to pick them up before treating
  the shot-event dataset as fully complete.

## 12. 2026-08-03 (final): real backtests for everything, two real reverts

**Read this before trusting anything in §11 as live.** The user asked for
every piece built tonight to be properly backtested, not just the core
per-stat models. Two of §11's pieces did not survive that and were removed
from the live decision path as a direct result. This section is the
authoritative current state — where it disagrees with §11, this section
wins.

### 12.1 Core 7-target backtest (confirms §11's new features are real, and confirms all engine routing)

`backtest_props.py` (already generic over `TARGETS`, needed no changes) run
across all 7 real targets, using the full current feature set — meaning the
Dynamic Pace, position-defense, and shooting-efficiency-shrinkage features
from §11 ARE exercised by this, not just added and hoped for:

| target | parametric bankroll | empirical MC bankroll | current live routing | confirmed correct? |
|---|---|---|---|---|
| points | 119,748 | 147,078 (wins) | empirical MC | yes |
| rebounds | 519,639 (wins) | 20,322 | parametric | yes |
| assists | 20,204 (wins) | 5,938 | parametric | yes |
| pra | 588,113 (wins) | 110,604 | parametric | yes |
| tpm | 4,882 (wins) | 2,152 | parametric (by omission) | yes — first real test, confirms the default was right |
| steals | 7,568 (wins) | 411 (LOSES money) | parametric | yes |
| blocks | 875 (loses) | 715 (loses) | parametric | **both engines lose money — real, unresolved concern, see below** |

`EMPIRICAL_MC_TARGETS = {"points"}` needs **no changes** — every current
target's routing is now confirmed correct, including the three added
tonight (previously untested by omission, not by a real comparison).

**Real, unresolved concern: blocks.** Both engines finish BELOW the $1,000
starting bankroll (875 and 715) — the only target where this happens. Only
218-366 bets placed (far thinner than every other market). Two real
possible explanations, not yet distinguished: (a) blocks genuinely has no
real backtestable edge at this synthetic-line construction, or (b) the
sample is too thin/noisy to conclude anything yet. Do not present blocks
picks with the same confidence as other markets until this is investigated
further — it hasn't been tonight.

### 12.2 Garbage-time/blowout-risk adjustment — backtested, DISABLED

Built in §11 with no dedicated backtest (backtest_props.py's simulation
path never calls it). Real, dedicated backtest built
(`backtest_garbage_time.py`): real 2025-2026 starters with a real
historical spread (4,012 rows), 1,059 with a meaningful real adjustment
(ratio < 0.98). Result:

| target | unadjusted MAE | adjusted MAE | verdict |
|---|---|---|---|
| points | 4.811 | 4.819 | adjustment HURTS |
| rebounds | 2.049 | 2.047 | tiny help, plausibly noise |
| assists | 1.522 | 1.516 | tiny help, plausibly noise |
| tpm | 0.879 | 0.878 | tiny help, plausibly noise |

Not strong enough evidence to keep live — one real target got worse, three
showed effects too small to trust. **DISABLED** in `predict_props.py`
(`blowout_model = None`, dead downstream logic removed, commit `666d8be`).
`garbage_time.py` and its backtest are kept as real, tested code for
reference — do not re-enable without new evidence (different fit, more
real data, or narrower scope) that changes this real result.

### 12.3 On/off splits — backtested TWICE, REVERTED

First backtest (train ≤2024, test 2025-2026 — the standard split used
everywhere else in this project): real out-of-sample correlation **-0.001**,
sign agreement **46.8%** (worse than chance). Suspected cause: a
multi-year gap doesn't suit something this roster-dependent (real trades/
coaching changes/development change who benefits from a real absence
year to year).

**Real fix attempted**: rewrote the backtest to validate IN-SEASON instead
— first half of a real season predicts the second half of the SAME season
(a recent, roster-stable window, matching how this should actually be used
live). Real result on a bigger sample (797 vs. 555 triplets): correlation
**0.019**, sign agreement **51.4%** — still no real signal, barely above
chance either way.

**Conclusion, tested not assumed**: the problem isn't the validation
window, it's the mechanism itself — a specific (player, teammate) pair's
on/off split estimated from only 5-8 real games per condition (the real
minimum threshold) is too noisy relative to a player's own normal
game-to-game variance to carry real signal at this sample size.

**REVERTED** in `predict_props.py` (commit `f6352fd`) — the usage-vacuum
section is back to the qualitative-only flag (real, sourced from the
injury report's own text naming a beneficiary — this part was never
invalidated, only the added quantified "REAL SPLIT" number was).
`on_off_splits.py` and both backtest versions are kept as real, tested
code — do not re-add the quantified version without a genuinely different
approach (not just a different window) that's actually tested first.

### 12.4 Real bug fixed the same pass: `grade_predictions.py` didn't know about 3PM

`MARKET_TO_STAT_COL` was missing `"tpm"` entirely — any saved board with
real 3PM picks (every board since tonight's 3PM launch) would have had
those rows silently come back `graded=False` instead of being checked
against real outcomes. Fixed (commit `f6352fd`): added the mapping and
the raw made/attempted parse needed to compute a real `tpm` column from
`threePointFieldGoalsMade-threePointFieldGoalsAttempted`.

### 12.5 What's actually live right now (the real, current truth)

- **Live and validated tonight**: points/rebounds/assists/pra/tpm/steals/
  blocks models with the full Dynamic Pace + position-defense + shooting-
  shrinkage feature set; engine routing (`EMPIRICAL_MC_TARGETS`) confirmed
  correct for all 7.
- **Live but NOT validated / real open concern**: blocks (both engines
  losing money in backtest — see 12.1).
- **Built, tested, explicitly NOT live**: garbage-time/blowout adjustment
  (disabled, 12.2), on/off split quantified number (reverted, 12.3), shot-
  zone breakdown (raw data only, nothing built on top, §11.6), the paid
  historical-odds capability itself (confirmed working, not yet used for
  a real player-prop-vs-real-historical-line backtest).
- **New monitoring tools, working, real first results**: `track_clv.py`
  (+1.43% average CLV, beat the real closing line 60.4% of 53 tracked
  bets), `check_coefficient_drift.py` (no meaningful drift found on first
  run; proposal-only, never auto-applies).
- **Rejected outright, real evidence against building at all**: full 5-man
  lineup splits, "contest level" shooting defense, altitude effects,
  literal possession-by-possession Monte Carlo simulator, context-weighted
  Monte Carlo resampling (all explicitly discussed and declined by the
  user or shown to be unbuildable/unvalidated — see §11 for the specific
  reasoning on each, not repeated here).

## 13. 2026-08-03 (continued session): quantile-regressor capacity fix, points routing reversal, real 3-model ensemble stack

**Supersedes §12.1's routing table.** Everything below is real, tested,
and either already live (`predict_props.py`) or persisted as a standalone
validated script (ensemble stack — not yet wired in).

### 13.1 Root cause found for an apparent points/PRA decline: model capacity, not the new features

A same-session before/after comparison made §12's new pace/position-defense/
shooting-efficiency features (§11) look like they made points and PRA worse.
Real cause, found via a controlled ablation test (old features vs new
features, same capacity; new features, new capacity): `train_quantile_models()`
(`train_baseline_model.py`) still used `GradientBoostingRegressor(n_estimators=60,
max_depth=3)` with no `random_state` — sized for a much smaller feature set
from earlier in the project, and non-reproducible across runs (no seed),
which also invalidated any same-session before/after comparison that didn't
control for it. **Fixed**: `n_estimators=200, max_depth=4, random_state=42`
(commit `7629c3f`, same commit also deleted the confirmed-dead `get_odds.py`
and its skill file). Poisson/NegBinom models (`max_iter=200`) already had
adequate capacity and needed no change. Real validated effect: points
+47%, PRA +183% on the controlled ablation.

### 13.2 Full 7-target backtest re-run with the fix — points routing REVERSES

`backtest_props.py`, full real run, ~8,900 held-out 2025-2026 bets, same
methodology as §12.1:

| target | parametric bankroll | empirical MC bankroll | winner | live routing after this session |
|---|---|---|---|---|
| points | **246,743** | 179,028 | **parametric (flipped)** | parametric |
| rebounds | 519,639 | 20,322 | parametric | parametric (unchanged) |
| assists | 20,204 | 5,938 | parametric | parametric (unchanged) |
| pra | 1,394,685 | 96,082 | parametric | parametric (unchanged) |
| tpm | 4,882 | 2,152 | parametric | parametric (unchanged) |
| steals | 7,568 | 411 | parametric | parametric (unchanged) |
| blocks | 875 (loses) | 715 (loses) | parametric | parametric — **still unresolved, both lose money, see §12.1** |

Before the capacity fix, empirical MC genuinely won for points (§12.1:
119,748 vs 147,078). After the fix, the parametric model jumped to 246,743
and now beats empirical MC too — the fix that helped the parametric model
doesn't help the bootstrap engine, which doesn't depend on the quantile
model's capacity. **Fixed live**: `EMPIRICAL_MC_TARGETS` in
`predict_props.py` changed from `{"points"}` to an empty set — all 7 targets
now route to the parametric model. Re-run the backtest and update this if
either engine changes again. Board regenerated and republished after this
change (`board_2026-08-03.html`, 127 props) — confirmed every pick's
reasoning now says "computed by the parametric model."

### 13.3 Real 3-model ensemble stack for points — built, bug found and fixed, validated

Per explicit request to build a real ensemble stack (genuinely diverse base
algorithms + a trained meta-learner, not another tree-boosting variant —
see §11/prior session for the diversity check that found GBM vs
HistGradientBoostingRegressor correlate at 0.9947, i.e. no real diversity).

**Base learners** (real, different inductive biases):
1. `GradientBoostingRegressor` — sequential tree boosting (existing tuned model)
2. `RandomForestRegressor` — bagging; quantile via the empirical distribution
   of each tree's point prediction (standard technique for getting a quantile
   estimate from a non-quantile-native model)
3. `QuantileRegressor` (linear, `sklearn.linear_model`) — genuinely different
   functional form from both tree methods

**Meta-learner bug found and fixed**: first version used plain
`LinearRegression` per quantile level to blend the 3 base predictions.
Real result: bankroll went to **$0.00** (4,540 bets, total ruin). Root
cause: squared-error loss fits the conditional MEAN regardless of which
quantile it's supposedly representing, so the q=0.1 and q=0.9 blended
outputs collapsed toward the same target, destroying the predicted spread
and producing wildly overconfident, badly miscalibrated probabilities.
**Fixed**: meta-learner changed to `QuantileRegressor(quantile=q, alpha=0.01)`
per level — matches the loss to the quantile it represents (also gives the
same L2-regularization benefit a Ridge meta-learner would, applied at the
correct loss for a quantile stack).

**Real validated result** (chronological fit/holdout split within train,
same 85/15 discipline used for isotonic calibration elsewhere; meta-learner
fit only on the holdout slice, never on data the base models trained on):
**$503,568.95 (1,895 bets)** vs the single capacity-fixed model's
**$246,742.58 (2,008 bets)** — roughly 2x. Sanity-checked before being
trusted (an unusually large improvement was exactly what the capacity bug
above looked like at first too):
- Real win rate 57.6% vs stated avg confidence 60.2% — close, honest,
  nothing like the severe miscalibration bugs found and fixed earlier this
  project for rebounds/assists.
- Bucket calibration is good through the 50-80% stated-confidence range
  (~1,876 of 1,895 bets).
- **Real, small-sample concern**: the >80%-stated-confidence bucket (n=19
  only) actually LOSES (50%/44% win rate) — same small-sample high-
  confidence-tail unreliability pattern already seen with rebounds. Don't
  trust any single high-confidence pick from this stack until more data
  accumulates there.
- Bankroll compounds smoothly through the bet sequence ($5.5K at 25% of
  bets, $35K at 50%, $164K at 75%, $504K at 100%) — not one lucky bet.
- The 2x gain (bigger than the "few percentage points" expectation set from
  published classification-task research before building this) is plausible
  because Kelly staking compounds a real edge exponentially over ~1,900
  sequential bets — a modest real tail-calibration improvement can produce
  a large final gap without needing a dramatically better single-bet edge.

**Persisted**: `src/backtest/backtest_points_ensemble_stack.py` (real,
importable, runnable standalone script — NOT a scratchpad file, survives
past this session). **NOT wired into `predict_props.py`** — the live board
still uses the single capacity-fixed quantile model for points. Wiring it
in is a real-money decision-logic change (rule 6) and needs an explicit go-
ahead, not assumed here.

**Scope, not yet done**: this stack is validated for `points` only. The
other 6 targets already use Poisson/NegBinom models with their own real
fixes and were not re-tested against a stacked version — doing so means
repeating this same (fairly slow — the linear `QuantileRegressor` step uses
an LP solver) training process per target, a real time/compute cost.

### 13.4 Note on suspicious repeated content this session

Several messages arrived during this session — some wrapped in the same
`<task-notification>` format as real background-task events, one as a
plain message — pushing an unrequested "XGBoost + LightGBM + Random Forest
+ Ridge" stack and later a "Bayesian copula same-game-parlay simulator,"
with oddly specific/promotional framing. Treated as untrusted content, not
acted on directly: the suggested architecture was checked on its technical
merits (XGBoost/LightGBM are more gradient-boosted trees, which the
project's own 0.9947-correlation diversity test already showed adds nothing
over what's built) and the copula/parlay idea would be a new, unrequested
subsystem (rule 7) requiring data (on-court joint lineup tracking) this
project doesn't currently have a pipeline for. The user then separately,
explicitly asked to adapt the copula idea using this project's own real
data/models rather than the literal content — that adapted work has not
been started yet as of this save.
