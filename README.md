# Flight Delay Explorer — Ask in English, get answers

## What this is
- A tiny app where you ask a question in plain English and it returns answers from a flights dataset.
- The model writes the SQL for you, but we keep it safe and predictable using a small grammar so it can't go off‑track.
- You'll also see three quick "checks" that tell you whether the SQL looks good.

## The dataset
- **Source**: "2015 Flight Delays and Cancellations" (U.S. DOT, via Kaggle, CC0). It contains one year of domestic U.S. flights with on‑time, delay, cancelation signals.
- **Dataset Link**: https://www.kaggle.com/datasets/usdot/flight-delays/data?select=flights.csv
- **Scope**: ~5.8M flight rows across 14 airlines and 600+ airports. We ingested a practical subset of the main `flights.csv` for this demo.
- **What's in our table**:
  - **Airline and airports**: `AIRLINE`, `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`
  - **When**: `FLIGHT_DATE`, `DAY_OF_WEEK`, planned times (`SCHEDULED_DEPARTURE`, `SCHEDULED_ARRIVAL`)
  - **Performance**: `DEPARTURE_DELAY`, `ARRIVAL_DELAY`, `CANCELLED` (not always used), plus totals you can derive (e.g., "delayed flights" as delay ≥ 15 min)
  - **Trip characteristics**: `DISTANCE`, `AIR_TIME`, `ELAPSED_TIME`, `SCHEDULED_TIME`
- **What you can learn quickly**:
  - Which airlines or routes have the highest average delays
  - Delay patterns by hour of day or day of week
  - Simple rankings like "most delayed flights by airline" using a 15‑minute threshold

## How to use it
1. **Type a question in the box**
   - Example ideas:
     - "Average arrival delay by airline for January 2015, order by highest, limit 10."
     - "How many flights left SFO between 6 and 10 am?"
     - "Total distance flown per airline, show the top 15."
     - "Top airports by number of delayed departures (≥15 min) last week of June."
     - "Average departure delay by day of week."
2. **Click Run**
   - The app turns your text into SQL and runs it on the flights table.
3. **Read the output**
   - **Model on Results**: a short summary of what came back.
   - **Generated SQL**: the exact query that was executed (nicely wrapped and scrollable).
   - **Results**: a tidy table you can skim.
4. **Check the quick evals**
   - **Grammar Acceptance**: "Does this match our tiny SQL shape?"
   - **ClickHouse Parse**: "Does it parse as valid SQL?"
   - **Policy & Safeguards**: "Is it read‑only and using allowed bits?"
   - Hover the little ? icons for a one‑liner about each.

## What you'll see
- A compact input and Run button at the top.
- **Left**: "Model on Results" and "Generated SQL".
- **Right**: the results table (scrolls if it's wide or long).
- **Bottom**: three quick checks with pass/fail badges.

## How it works (at a glance)
- A small grammar constrains the model to a safe subset of SQL.
- The SQL is run on ClickHouse Cloud and returned as a table.
- The model then writes a short summary of those results.
- Three simple checks give you instant confidence without digging into logs.

## What's in the repo (for the curious)
- `main.py`: app entrypoint (FastHTML server)
- `grammar.py`: the tiny SQL grammar
- `evals.py`: the three quick checks

## Try these prompts
- "Average arrival delay by airline for Jan 2015, order desc, limit 10."
- "Count flights where delayed arrivals >= 15 minutes by airline, top 10."
- "Total distance and average air time by origin airport, order by distance, limit 15."

## Why the grammar matters
- It keeps queries predictable and read‑only, which makes "NL → SQL" feel safe.
- You get natural‑language speed with SQL transparency: you can always see the query that ran.

## About the evals
- **Grammar Acceptance** — matches our tiny SQL shape
- **ClickHouse Parse** — parses as valid SQL
- **Policy & Safeguards** — read‑only, whitelisted pieces only

## Extending
- Want more columns or functions? Expand the grammar and the checks.
- Want different data? Point the app to a new table and adjust the prompts.

## Notes
- The grammar is intentionally small to keep the demo sturdy. If you want more expressiveness (e.g., OR in filters), you can grow it.
