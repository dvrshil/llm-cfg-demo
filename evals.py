from __future__ import annotations
from typing import Tuple, List, Set

from lark import Lark
from lark.exceptions import LarkError
from sqlglot import parse_one, errors as sg_errors, exp

# Reuse the grammar from grammar.py
from grammar import clickhouse_flights_grammar


# ---------- 1) Lark syntax acceptance ----------
_grammar_parser: Lark | None = None


def _get_parser() -> Lark:
    global _grammar_parser
    if _grammar_parser is None:
        _grammar_parser = Lark(
            clickhouse_flights_grammar, start="start", parser="lalr", lexer="basic"
        )
    return _grammar_parser


def grammar_accepts(sql: str) -> Tuple[bool, str]:
    try:
        _get_parser().parse(sql)
        return True, ""
    except LarkError as e:
        return False, f"Lark rejection: {e}"


# ---------- 2) General SQL syntax with sqlglot (ClickHouse) ----------
def is_valid_clickhouse_sql(sql: str) -> Tuple[bool, str]:
    """True if sqlglot can parse the query under the ClickHouse dialect."""
    try:
        parse_one(sql, read="clickhouse")
        return True, ""
    except sg_errors.ParseError as e:
        return False, f"sqlglot ParseError: {e}"


# ---------- 3) Policy / safety checks (AST-based) ----------
ALLOWED_COLS: Set[str] = {
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "DAY_OF_WEEK",
    "FLIGHT_DATE",
    "DEPARTURE_DELAY",
    "ARRIVAL_DELAY",
    "DISTANCE",
    "AIR_TIME",
    "ELAPSED_TIME",
    "SCHEDULED_TIME",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_ARRIVAL",
}

ALLOWED_FUNCS: Set[str] = {
    # aggregates
    "AVG",
    "SUM",
    "MIN",
    "MAX",
    "COUNT",
    "COUNTIF",
    "QUANTILETDIGEST",
    "QUANTILEEXACT",
    # scalar/derived
    "INTDIV",
    "ABS",
    "*",
}

FORBIDDEN_NODES = (
    exp.Update,
    exp.Delete,
    exp.Insert,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Join,
    exp.Union,
    exp.With,  # forbid CTEs
    exp.Window,  # forbid window functions
    exp.Having,  # keep it simple; allow later if desired
    exp.Set,  # SET statements
)


def policy_check(sql: str) -> Tuple[bool, List[str]]:
    """
    Enforces read-only, single-SELECT, no joins/unions, only whitelisted cols/funcs.
    Returns (ok, problems[])
    """
    problems: List[str] = []
    try:
        tree = parse_one(sql, read="clickhouse")
    except sg_errors.ParseError as e:
        return False, [f"Not parseable for policy check: {e}"]

    # Must be a SELECT
    if not isinstance(tree, exp.Select):
        top_select = tree.find(exp.Select)
        if not top_select:
            problems.append("Query is not a SELECT.")
            return False, problems
        tree = top_select

    # Forbid risky constructs
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            problems.append(f"Forbidden clause: {type(node).__name__}")

    # Columns whitelist
    for col in tree.find_all(exp.Column):
        name = (col.alias_or_name or "").upper()
        if name and name not in ALLOWED_COLS:
            problems.append(f"Unknown column: {name}")

    # Functions whitelist
    for f in tree.find_all(exp.Func):
        fname = f.name.upper() if hasattr(f, "name") and f.name else None
        if fname and fname not in ALLOWED_FUNCS:
            problems.append(f"Function not allowed: {fname}")

    # Subqueries?
    if tree.find(exp.Subquery):
        problems.append("Subqueries are not allowed.")

    return (len(problems) == 0), problems


# ---------- Example NL queries for evals and demos ----------
# These prompts exercise different dimensions of the dataset while
# staying within the app's grammar: SELECT-only, GROUP BY/ORDER BY,
# hour filters via scheduled dep/arr hours, and 15-minute delay thresholds.
EVAL_NL_QUERIES: List[str] = [
    # Airline‑level summaries
    "Which airlines had the worst delays when flights arrived in January 2015? Show me the top ten.",
    "Tell me about airlines with the worst departure delays and how many of their flights were really late (15+ minutes).",
    "Which airlines flew the most miles across 2015?",
    "Which airlines were best at getting flights to arrive on time in Q2 2015?",
    # Airport‑level summaries
    "Which airports are worst for departure delays when flights leave from there?",
    "Which airports see the most really late arrivals (15+ minutes)?",
    "Which airports had the best departure delays in July 2015?",
    # Route breakdowns (origin → destination)
    "What are the worst routes for arrival delays? Show me flights from one airport to another.",
    "How far do different routes go and how long do they take in the air? Show me the longest routes.",
    # Day‑of‑week patterns
    "How bad are departure delays on different days of the week?",
    "How many flights arrived really late (15+ minutes) on each day of the week?",
    # Hour‑of‑day windows
    "During morning rush hour (6-10am), which airlines have the worst departure delays?",
    "During evening rush (5-8pm), which airports have the most really late departures (15+ minutes)?",
    # Date ranges
    "How did airlines do with arrival delays during June 2015?",
    "Which airports had flights covering the most distance in the last 12 days of December 2015?",
    # Month‑specific comparisons
    "How did airlines perform with departure delays in March 2015 — who was best and worst?",
    "Which airports had the most really late arrivals (15+ minutes) in November 2015?",
    # Mix of aggregates
    "For each airline, show me their typical, best, and worst arrival delays. Order by typical delay.",
    "For each airport, show me their typical, best, and worst departure delays. Order by typical delay.",
    # Speed proxy (mph)
    "Which airlines had the fastest flights on average?",
    "How fast were flights from different airports in June 2015?",
    # Peak times exploration
    "How bad are arrival delays in the evening when flights are scheduled to land between 6-11pm?",
    "How many flights arrived really late (15+ minutes) in the morning rush (7-9am) for each hour?",
    # Specific carriers or airports
    "For Southwest (WN), how do departure delays vary by day of the week?",
    "From San Francisco (SFO), which destinations have the worst arrival delays?",
    "To JFK airport, which origin airports send flights covering the most distance?",
    # Holiday windows
    "How did airlines do with departure delays around Thanksgiving week (Nov 20-30, 2015)?",
    "Which airports had the most really late arrivals (15+ minutes) during Christmas week (Dec 20-27, 2015)?",
    # Morning vs evening
    "How do airlines perform with departure delays in the morning (6-9am)?",
    "How do airlines perform with departure delays in the evening (5-8pm)?",
    # Route leaders
    "Which airports have flights that cover the most total distance when departing from there?",
    "Which airports receive flights that cover the most total distance when arriving there?",
    # Delay distributions via percentiles
    "For each airline, what's the 90th percentile arrival delay — who has the worst outliers?",
    "What's the 95th percentile departure delay for each airport?",
    # Weekday patterns by airport
    "For flights going to Chicago O'Hare (ORD), how do arrival delays vary by day of week?",
    "For flights leaving Atlanta (ATL), how do departure delays vary by day of week?",
    # Combined metrics
    "For each destination airport, what's the typical arrival delay and how many flights arrive really late (15+ minutes)?",
    "In June 2015, for each airline, what was the typical departure delay and how many flights left really late (15+ minutes)?",
    # Short vs long trips
    "For each airline, how are arrival delays on their longer flights?",
    "For each destination airport, how are arrival delays on longer flights coming in?",
    # Simple sanity checks
    "About how many flights are in this dataset?",
    "What's the typical arrival delay overall?",
]


def sample_queries() -> List[str]:
    """Return example natural language prompts for demo/evals."""
    return EVAL_NL_QUERIES
