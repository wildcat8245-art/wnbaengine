---
name: draftkings-odds
description: >-
  Fetch live betting odds from DraftKings' sportsbook API as a pandas
  DataFrame. Use whenever the user wants current DraftKings odds, moneylines,
  spreads, or totals for a league (WNBA by default) — e.g. "get the DraftKings
  odds", "what are the WNBA lines", "pull live odds into a table/CSV". Runs the
  get_odds.py script in this repository.
---

# DraftKings Odds Fetcher

This skill fetches live odds from DraftKings using the `get_odds.py` script at
the repository root. The script queries DraftKings' undocumented sportsbook
JSON API, flattens the nested markets/outcomes, and returns a tidy pandas
DataFrame (one row per betting outcome).

## Prerequisites

Install the two runtime dependencies if they are not already present:

```bash
pip install requests pandas
```

## Usage

Run the script from the repository root.

- Default (WNBA odds, printed as a table):

  ```bash
  python get_odds.py
  ```

- A specific DraftKings event group id (each league/competition has one):

  ```bash
  python get_odds.py --event-group 42648
  ```

- Save the odds to a CSV file instead of printing:

  ```bash
  python get_odds.py --output odds.csv
  ```

To use it programmatically, import `get_odds`:

```python
from get_odds import get_odds

df = get_odds()            # WNBA
df = get_odds("42648")     # another event group
```

## Output columns

Each row is a single outcome with these fields:

| Column          | Description                                        |
| --------------- | -------------------------------------------------- |
| `event_id`      | DraftKings event identifier                        |
| `event_name`    | Human-readable matchup name                         |
| `start_date`    | Event start time (ISO 8601)                        |
| `category`      | Offer category (e.g. Game Lines)                   |
| `market`        | Market/subcategory (e.g. Moneyline, Spread, Total) |
| `label`         | Offer label                                        |
| `outcome`       | Outcome label (team name, Over/Under, etc.)        |
| `line`          | Point spread or total, when applicable             |
| `odds_american` | American odds (e.g. -110, +150)                    |
| `odds_decimal`  | Decimal odds                                       |

## Finding an event group id

The default event group is `94682` (WNBA). To find another league's id, browse
to that league on `sportsbook.draftkings.com` and read the numeric id from the
URL (`.../leagues/<sport>/<event-group-id>`).

## Notes

- The DraftKings API is undocumented and can change without notice; if parsing
  returns an empty DataFrame, the payload structure may have shifted.
- Use for personal/educational purposes only and respect DraftKings' terms of
  service and rate limits. A browser-like User-Agent is already set in the
  script.
