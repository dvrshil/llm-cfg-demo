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

