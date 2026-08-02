# Rules for working on this project

These are binding requirements, not suggestions. Follow them every session,
not just when reminded.

## 1. Backtest before live, every time

Any change to what `predict_props.py` actually decides (which engine drives
a pick, staking logic, thresholds, routing) MUST be run through
`backtest_props.py` in the same session and the results reported, before
telling the user the change is ready to bet on. Do not ship an unvalidated
change and check it later — that already happened once (2026-08-01/08-02):
an engine change went live 37 minutes after being written, wasn't backtested
for a full day, and turned out to lose badly on 3 of 4 markets. Never repeat
that sequence.

## 2. The live injury check and usage-vacuum check are permanent, not optional

`predict_props.py` calls `src/data/injuries_client.py` on every run. This is
part of the system, not an add-on step to remember to run separately. If you
touch `predict_props.py`, do not remove or bypass this call. If the injury
API call fails, the board must say so loudly (it already does) — never
present an unchecked board as if it were checked.

## 3. Never write an unverified claim into any file as if it were settled fact

Before writing "the user decided X," "explicitly rejected X," "the design is
based on X," or similar framing into `SESSION_NOTES.md`, a commit message,
or anywhere else — you need an actual quote from the current conversation.
If you don't have one, don't write it as settled. This project's own history
has two confirmed real instances of this failure:
- A false "Monte Carlo simulation... rejected up front" note that carried no
  user quote and wasn't true.
- A false "adapted from a real academic paper" framing, repeated across
  code comments and `SESSION_NOTES.md`, that the user never asked for and
  states directly is not true.
Both caused real, justified anger and both had to be found and stripped out
by hand. Do not add a third.

## 4. Describe only what the code does — no invented provenance

Comments and notes explain mechanism (what the code does and why, in terms
of this codebase's own real bugs/fixes/data), not narrative about where the
idea supposedly came from. Do not attribute this system's design to a paper,
citation, or external methodology unless the user says so explicitly in the
current conversation.

## 5. Verify current state before reporting anything

Run `git status` and `git diff` at the start of any session touching this
repo — do not assume the last commit reflects the current code. Multiple
sessions have left real, working changes uncommitted for a full day. Check
live data (API calls, file timestamps) over trusting old notes, including
this file's own claims about what's already built — verify, don't assume.

## 6. Ask before executing changes to real-money decision logic

Changes to what `predict_props.py` recommends are a business decision, not
a routine implementation choice. State the proposed change and wait for an
actual "yes" — never write "I'm going to make that change now" and execute
it in the same turn, even when the evidence clearly supports it. Ordinary
implementation details (pipeline structure, variable names, which library)
don't need this — this rule is specifically for anything that changes which
side/stake a real bet gets.
