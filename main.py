import os
import asyncio
import re
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from fasthtml.common import *
from typing import Optional, Tuple, Any
from monsterui.all import *
from grammar import clickhouse_flights_grammar, sql_lark_cfg_tool
from openai import AsyncAzureOpenAI
from evals import grammar_accepts, is_valid_clickhouse_sql, policy_check, sample_queries
import random

load_dotenv()

# ClickHouse Cloud API credentials
key_id = os.environ.get("CH_KEY_ID")
key_secret = os.environ.get("CH_KEY_SECRET")
service_id = "88ac54be-2166-445d-9172-dc3173309069"


def run_clickhouse_query(sql_query: str, format_type: str = "CSVWithNames") -> str:
    """Execute SQL query against ClickHouse Cloud using REST API"""
    url = f"https://queries.clickhouse.cloud/service/{service_id}/run"
    response = requests.post(
        url,
        auth=(key_id, key_secret),
        headers={"Content-Type": "application/json"},
        params={"format": format_type},
        json={"sql": sql_query},
        timeout=45,
    )
    response.raise_for_status()
    return response.text


client = AsyncAzureOpenAI(
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
)


async def generate_sql_from_nl(
    natural_language_query: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Use Azure OpenAI with a grammar tool to produce a safe SQL string and optional model text."""

    prompt = (
        "You translate natural language into valid ClickHouse SQL over table flights_df.\n"
        "Constraints (must follow): SELECT-only; FROM flights_df; WHERE supports AND only; GROUP BY; ORDER BY; LIMIT.\n"
        "Columns available: AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, FLIGHT_DATE, DAY_OF_WEEK, "
        "DEPARTURE_DELAY, ARRIVAL_DELAY, DISTANCE, AIR_TIME, ELAPSED_TIME, SCHEDULED_TIME; "
        "SCHEDULED_DEPARTURE and SCHEDULED_ARRIVAL are HHMM integers used via intDiv(*,100) to get the hour.\n"
        "Units: delays/times are minutes; distances are miles; DAY_OF_WEEK is 1..7.\n"
        "Derived expressions permitted: intDiv(SCHEDULED_DEPARTURE,100), intDiv(SCHEDULED_ARRIVAL,100), (DISTANCE*60/AIR_TIME).\n"
        "Aggregates permitted: avg,sum,min,max,count,countIf(cond),quantileTDigest/Exact(p)(field).\n"
        "If filtering by hour, use the intDiv-derived expressions or the provided BETWEEN hour clauses.\n"
        "Use only constructs allowed by the provided grammar. End every statement with a semicolon.\n"
        "Only return a valid SQL statement during the first turn of the conversation, and nothing else.\n"
        "User request: " + natural_language_query
    )

    resp = await client.responses.create(
        model="gpt-5-mini",
        input=prompt,
        text={"format": {"type": "text"}},
        tools=[sql_lark_cfg_tool],
        parallel_tool_calls=False,
        reasoning={"effort": "low"},
    )

    sql_query: Optional[str] = None
    llm_text: Optional[str] = None

    # Best-effort extraction of tool call SQL and any text response
    try:
        for item in getattr(resp, "output", []) or []:
            typ = getattr(item, "type", None)
            if hasattr(item, "input") and isinstance(getattr(item, "input"), str):
                candidate = getattr(item, "input")
                if "SELECT" in candidate.upper():
                    # Clean up the candidate to remove any metadata
                    candidate = candidate.strip()

                    # Use regex to extract just the SQL part
                    sql_match = re.search(
                        r"(SELECT\s+.*?;)", candidate, re.IGNORECASE | re.DOTALL
                    )
                    if sql_match:
                        sql_query = sql_match.group(1).strip()
                    else:
                        # Fallback: filter out lines that look like debug/metadata
                        lines = candidate.split("\n")
                        sql_lines = []
                        for line in lines:
                            line = line.strip()
                            # More comprehensive filtering
                            if (
                                line
                                and not any(
                                    keyword in line
                                    for keyword in [
                                        "type=",
                                        "logprobs=",
                                        "annotations=",
                                        "ResponseOutputText",
                                        "text=",
                                        "output_text",
                                        "logprobs=None",
                                    ]
                                )
                                and not line.startswith("[")
                                and not line.startswith("(")
                            ):
                                sql_lines.append(line)
                        if sql_lines:
                            sql_query = " ".join(sql_lines)
            if typ == "message" and hasattr(item, "content"):
                try:
                    llm_text = str(item.content)
                except Exception:
                    pass
        if sql_query is None:
            # Fallback to the common pattern used earlier
            try:
                sql_query = resp.output[1].input  # type: ignore[attr-defined]
            except Exception:
                pass
        # Capture some model text even if not present in message items
        if (
            llm_text is None
            and hasattr(resp, "output_text")
            and isinstance(resp.output_text, str)
        ):
            llm_text = resp.output_text
        # Fallback: extract SQL from plain text if tools didn't surface it
        if sql_query is None and isinstance(llm_text, str):
            m = re.search(r"SELECT\s", llm_text, flags=re.IGNORECASE)
            if m:
                # Extract everything from SELECT to the end, then clean it up
                cand = llm_text[m.start() :].strip()

                # Use regex to extract just the SQL part first
                sql_match = re.search(
                    r"(SELECT\s+.*?;)", cand, re.IGNORECASE | re.DOTALL
                )
                if sql_match:
                    sql_query = sql_match.group(1).strip()
                else:
                    # Try to find the end of the SQL statement
                    # Look for semicolon or end of meaningful SQL content
                    lines = cand.split("\n")
                    sql_lines = []
                    for line in lines:
                        line = line.strip()
                        # Skip empty lines and lines that look like metadata/debug info
                        if (
                            not line
                            or any(
                                keyword in line
                                for keyword in [
                                    "type=",
                                    "logprobs=",
                                    "annotations=",
                                    "ResponseOutputText",
                                    "text=",
                                    "output_text",
                                    "logprobs=None",
                                ]
                            )
                            or line.startswith("[")
                            or line.startswith("(")
                        ):
                            if sql_lines:  # If we already have SQL content, stop here
                                break
                            continue
                        sql_lines.append(line)
                        # Stop if we hit a semicolon
                        if line.endswith(";"):
                            break

                    if sql_lines:
                        cand = " ".join(sql_lines)
                        if not cand.endswith(";"):
                            cand += ";"
                        sql_query = cand
    except Exception:
        raise

    if not sql_query:
        # No SQL extracted; still return any model text captured
        return None, llm_text
    sql_query = sql_query.strip()
    if not sql_query.endswith(";"):
        sql_query += ";"
    return sql_query, llm_text


def render_df_table(df: pd.DataFrame) -> Any:
    """Render a pandas DataFrame as HTML inside an FT Div without escaping."""
    head_df = df.head(200)
    table_html = head_df.to_html(index=False, border=0)
    return Div(NotStr(table_html), cls="table-scroll")


# ----- LLM follow-up with ClickHouse result -----
def _truncate_text(text: str, max_lines: int = 60, max_chars: int = 8000) -> str:
    """Trim large strings to keep LLM prompts small and predictable."""
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["... (truncated)"]
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[: max_chars - 15] + "\n...(truncated)"
    return out


async def llm_observe_clickhouse_result(
    nl_query: str, sql_query: str, csv_text: str
) -> Optional[str]:
    """Inform the LLM of the executed SQL and ClickHouse result, and get a short summary.

    Keeps grammar-based SQL generation separate (first call), then makes a simple text-only
    follow-up call so the model can "see" what ClickHouse returned. Returns a concise
    model summary/acknowledgment for display, or None if not available.
    """
    try:
        trimmed_csv = _truncate_text(csv_text or "")
        followup_instructions = (
            "Here's the result of executing a ClickHouse query. "
            "Do not generate any new SQL now, unless an error occurred. Provide a concise "
            "summary or 2-3 observations."
            "Only use bullets to format your response. Make the report as easy and interesting as possible."
            "Only when mentioning flight or airport names, include ID guide for them in your response at the bottom."
            "Use proper units behind numbers, they are in mins., miles or HHMM format"
            "Keep it under 100 words"
        )

        followup_input = (
            f"{followup_instructions}\n\n"
            f"User request:\n{nl_query}\n\n"
            f"Executed SQL:\n{sql_query}\n\n"
            f"ClickHouse CSVWithNames result (truncated):\n{trimmed_csv}\n"
        )

        resp = await client.responses.create(
            model="gpt-5-mini",
            input=followup_input,
            text={"format": {"type": "text"}},
            reasoning={"effort": "minimal"},
            parallel_tool_calls=False,
        )
        # Prefer output_text if present; otherwise scan items
        if hasattr(resp, "output_text") and isinstance(resp.output_text, str):
            return resp.output_text
        for item in getattr(resp, "output", []) or []:
            typ = getattr(item, "type", None)
            if typ == "message" and hasattr(item, "content"):
                try:
                    return str(item.content)
                except Exception:
                    pass
    except Exception:
        # Non-fatal for the UI; skip follow-up text on any error
        return None
    return None


# ----- FastHTML app -----
# Force dark mode for MonsterUI theme (disable light theme)
_theme_hdrs = Theme.blue.headers(mode="dark", highlightjs=True)
# App-specific minimal CSS to polish layout
_app_css = Style(
    """
    :root { color-scheme: dark; --surface: #0c0f13; --surface-2:#0f141a; --border:#222933; --shadow: 0 10px 24px rgba(0,0,0,.35); }
    /* Force the Titled() H1 to align with our container */
    body h1, h1, .container h1, main h1 {
        width: 100% !important;
        margin: 6rem auto .75rem auto !important; /* generous top space */
        letter-spacing: 1px !important;
        text-align: center !important; /* centered title */
        font-weight: 900 !important;
        font-size: clamp(36px, 5vw, 56px) !important;
        line-height: 1.1 !important;
        text-wrap: balance !important;
        display: block !important; 
        box-sizing: border-box !important;
    }
    .subtitle, div.subtitle {
        width: 100% !important;
        margin: 0 auto 1.5rem auto !important;
        text-align: center !important; /* centered subtitle */
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        opacity: .95 !important;
        text-wrap: balance !important;
        display: block !important;
        letter-spacing: 0.5px !important;
    }
    /* Additional ultra-specific selectors to override any framework styles */
    body > main > h1,
    body > div > h1,
    main > h1,
    .container > h1 {
        text-align: center !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    body > main > .subtitle,
    body > div > .subtitle,
    main > .subtitle,
    .container > .subtitle {
        text-align: center !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    .container-narrow { width: min(1200px, 100% - 48px); margin-inline: auto; }
    .page-pad { padding-block: 1.25rem 2rem; }
    .hero { margin-bottom: 4rem; }
    .hero h1 { margin-bottom: .25rem; }
    .lead { width: min(900px, 100% - 48px); margin: .25rem auto .75rem auto; font-size: 1.06rem; line-height: 1.6; text-wrap: balance; text-align: center; }
    .muted { opacity:.85; font-size: .95rem; }
    .panel { border: 1px solid var(--border); border-radius: 12px; padding: 1rem; background: var(--surface-2); box-shadow: var(--shadow); }
    .panel h3 { margin: 0 0 .5rem 0; font-size: 0.98rem; opacity: .95; }
    .query-grid { display:grid; grid-template-columns: 1fr auto; gap: .9rem; align-items: start; }
    .run-btn { display: flex; align-items: center; justify-content: center; border-radius: 12px; padding-inline: 1.25rem; height: 52px; font-weight: 650; font-size: 16px; }
    .query-grid textarea { width: 100%; height: 52px; min-height: 52px; max-height: 52px; resize: none; border-radius: 12px; border: 1px solid var(--border); background: var(--surface); color: inherit; padding: .9rem 1rem; }
    .query-grid textarea::placeholder { opacity: .7; font-style: italic; }
    .form-title { margin-bottom: 1rem; font-size: 1.1rem; font-weight: 600; color: inherit; }
    /* Suggestions */
    .suggestions-wrap { margin: 1rem 0 0 0; padding: 0; }
    .suggestions-compact { display:flex; align-items:center; gap:.75rem; flex-wrap:wrap; }
    .suggestion-label { font-size: .9rem; opacity: .8; font-weight: 500; white-space: nowrap; }
    .suggestions-inline { display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; }
    .chip { border:1px solid var(--border); border-radius:999px; padding:.3rem .6rem; background:var(--surface); cursor:pointer; font-size: .85rem; transition: all 0.2s ease; }
    .chip:hover { background: rgba(255,255,255,.06); border-color: rgba(255,255,255,.2); }
    .reload-btn-small { padding: .2rem; width: 24px; height: 24px; border-radius: 4px; display: flex; align-items: center; justify-content: center; opacity: .7; }
    .reload-btn-small:hover { background: rgba(255,255,255,.05); opacity: 1; }
    /* Loading indicators */
    .htmx-indicator { opacity: 0; transition: opacity 0.2s ease; }
    .run-btn.htmx-request .htmx-indicator { opacity: 1; }
    #results.htmx-request .htmx-indicator { opacity: 1; }
    /* Hide content during loading */
    #results.htmx-request > *:not(.loading-overlay) { opacity: 0.3; }
    .loading-overlay { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 10; }
    .results-grid { display:grid; grid-template-columns: 1fr; gap: 1.25rem; align-items:start; margin-top: 1rem; }
    .results-grid > * { width: 100%; }
    @media (min-width: 1100px) { .results-grid { grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr); } }
    .left-stack { display:grid; gap: 1rem; width: 100%; }
    .panel { box-sizing: border-box; width: 100%; }
    /* SQL viewer */
    .sql-wrap { position: relative; }
    .sql-actions { position: absolute; top: .4rem; right: .5rem; display:flex; gap:.5rem; z-index: 2; }
    .sql-view { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                font-size: .9rem; line-height: 1.35; white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere;
                background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: .8rem .95rem;
                max-height: 380px; overflow: auto; margin: 0; }
    /* Table polish + scroll */
    .table-scroll { width: 100%; overflow: auto; max-height: 420px; }
    #results table { min-width: 100%; width: max-content; border-collapse: collapse; white-space: nowrap; border: 1px solid var(--border); }
    #results thead th { position: sticky; top: 0; background: var(--surface-2); z-index: 1; text-align: left; }
    #results th, #results td { padding: .5rem .75rem; border: 1px solid var(--border); font-size: .92rem; text-align: left; }
    #results tr:hover td { background: rgba(255,255,255,.02); }
    .tight { margin-block: .5rem; }
    /* Placeholder phases inside results */
    #results .placeholder { display:block; margin: .25rem 0; }
    #results .placeholder-loading { display:none; }
    #results.htmx-request .placeholder-idle { display:none; }
    #results.htmx-request .placeholder-loading { display:block; }
    #results .dots:after { content: '…'; animation: dots 1.2s steps(4,end) infinite; }
    @keyframes dots { 0%,20%{content:''} 40%{content:'.'} 60%{content:'..'} 80%,100%{content:'...'} }
    /* Results and Evals styling */
    .results-section { margin-top: 1rem; }
    .results-section > h3 { margin: 0 0 0.5rem 0; }
    .evals-section { margin-top: 2.5rem; }
    .evals-section > h3 { margin: 0 0 1rem 0; }
    .eval-grid { display:grid; grid-template-columns: 1fr; gap: 2rem; }
    @media (min-width: 900px) { .eval-grid { grid-template-columns: repeat(3, 1fr); } }
    /* Stack vertically when there are multiple failures for better readability */
    .eval-grid.multiple-failures { grid-template-columns: 1fr !important; }
    .eval-tile { position: relative; width: 100%; max-width: 100%; box-sizing: border-box; overflow: hidden; }
    .status-pass { color: #32d296; font-weight: 600; }
    .status-fail { color: #ff6b6b; font-weight: 600; }
    .eval-title { display:flex; align-items:center; justify-content: space-between; gap: .75rem; margin-bottom: .35rem; }
    .badge { border:1px solid var(--border); border-radius: 6px; padding: .15rem .5rem; font-size: .78rem; font-weight: 650; line-height: 1.2; }
    .badge-pass { color:#32d296; background: rgba(50,210,150,.08); border-color: rgba(50,210,150,.35); }
    .badge-fail { color:#ff6b6b; background: rgba(255,107,107,.10); border-color: rgba(255,107,107,.40); }¸
    .eval-desc { margin: .15rem 0 .5rem 0; opacity: .9; }
    .eval-code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                font-size: .85rem; line-height: 1.3; white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere;
                background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: .6rem .75rem;
                max-height: 200px; width: 100%; max-width: 100%; overflow: auto; margin: .25rem 0; box-sizing: border-box; }
    /* Schema modal styling */
    .schema-modal { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); 
                   display: flex; align-items: center; justify-content: center; z-index: 1000; }
    .schema-content { background: var(--surface-2); border: 1px solid var(--border); border-radius: 12px; 
                     padding: 2rem; max-width: 90vw; max-height: 80vh; overflow: auto; position: relative; }
    .schema-table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
    .schema-table th, .schema-table td { padding: 0.75rem; text-align: left; border: 1px solid var(--border); }
    .schema-table th { background: var(--surface); font-weight: 600; }
    .schema-table tr:hover td { background: rgba(255,255,255,.02); }
    .schema-close { position: absolute; top: 1rem; right: 1rem; }
    """
)
hdrs = (*_theme_hdrs, MarkdownJS(), _app_css)
app, rt = fast_app(hdrs=hdrs)
setup_toasts(app)


def mk_layout(content: Any) -> Any:
    """Standard page layout wrapper (no duplicated titles)."""
    title = "Flight Delay Explorer"
    subtitle = "Ask in English → SQL → ClickHouse answers → we summarize"
    # Titled already provides <title> and <h1>
    return Titled(
        title,
        Div(
            subtitle,
            cls="muted tight subtitle",
        ),
        Div(content, cls="container-narrow page-pad"),
    )


def mk_form() -> Any:
    """Create a compact query form with inline run button and spinner."""
    example = (
        "Try: morning vs evening delays by airline, or top airports by delayed arrivals"
    )
    input_box = Textarea(
        id="nl_query",
        name="nl_query",
        rows=3,
        placeholder=f"e.g. {example}",
    )
    form = Form(
        hx_post=run.to(),
        hx_target="#results",
        hx_swap="outerHTML",
        hx_indicator="#results",
        # Allow button submit and throttled Enter from textarea
        hx_trigger="submit, keydown[key=='Enter'&&!shiftKey] from:#nl_query consume throttle:900ms",
        # Make extra presses impossible while a request is in flight
        hx_disabled_elt="#nl_query, .run-btn",
        cls="panel",
    )(
        H3("Ask the flights dataset", cls="form-title"),
        Div(
            input_box,
            Button(
                "Run",
                type="submit",
                cls=f"{ButtonT.primary} run-btn",
                hx_indicator="this",
            ),
            cls="query-grid",
        ),
        render_suggestions(),
    )
    return form


def _pick_suggestions(n: int = 4) -> list[str]:
    try:
        pool = sample_queries()
        return random.sample(pool, k=min(n, len(pool)))
    except Exception:
        return []


def _suggestion_chip(text: str) -> Any:
    return Span(
        text,
        cls="chip",
        data_suggestion=text,
        onclick=f"const t = document.getElementById('nl_query'); t.value = '{text}'; t.focus(); console.log('Filled with: {text}'); setTimeout(() => htmx.trigger(t.form, 'submit'), 100);",
    )


def render_suggestions() -> Any:
    suggestions_list = _pick_suggestions(1)
    items = [_suggestion_chip(s) for s in suggestions_list] if suggestions_list else []
    return Div(
        Div(
            Span("Try this example:", cls="suggestion-label"),
            Div(*items, cls="suggestions-inline"),
            Button(
                UkIcon("refresh-cw", height=14),
                cls=f"{ButtonT.ghost} reload-btn-small",
                hx_get=suggestions.to(),
                hx_target="#suggestions",
                hx_swap="outerHTML",
                title="Load another example",
                type="button",
                onclick="event.stopPropagation(); event.preventDefault();",
            ),
            cls="suggestions-compact",
        ),
        id="suggestions",
        cls="suggestions-wrap",
    )


@rt
def suggestions():
    return render_suggestions()


@rt
def show_schema():
    """Display the database schema in a modal."""
    schema_data = [
        ("AIRLINE", "String", "Airline code (e.g., AA, DL, UA)"),
        ("ORIGIN_AIRPORT", "String", "Origin airport code (e.g., LAX, JFK)"),
        ("DESTINATION_AIRPORT", "String", "Destination airport code"),
        ("FLIGHT_DATE", "Date", "Flight date (YYYY-MM-DD format)"),
        ("DAY_OF_WEEK", "Int64", "Day of week (1=Monday, 7=Sunday)"),
        ("SCHEDULED_DEPARTURE", "Int64", "Scheduled departure time (HHMM format)"),
        ("SCHEDULED_ARRIVAL", "Int64", "Scheduled arrival time (HHMM format)"),
        ("DEPARTURE_DELAY", "Float64", "Departure delay in minutes"),
        ("ARRIVAL_DELAY", "Float64", "Arrival delay in minutes"),
        ("DISTANCE", "Int64", "Flight distance in miles"),
        ("AIR_TIME", "Float64", "Flight time in minutes"),
        ("ELAPSED_TIME", "Float64", "Total elapsed time in minutes"),
        ("SCHEDULED_TIME", "Float64", "Scheduled flight time in minutes"),
    ]

    table_rows = []
    for field, field_type, description in schema_data:
        table_rows.append(
            Tr(
                Td(Strong(field)),
                Td(field_type, style="font-family: ui-monospace, monospace;"),
                Td(description, cls="muted"),
            )
        )

    modal_content = Div(
        Button(
            UkIcon("x", height=16),
            cls=f"{ButtonT.ghost} schema-close",
            hx_get="/",
            hx_target="body",
            hx_swap="outerHTML",
        ),
        H2("Database Schema: flights_df"),
        P("U.S. flight delay data from 2015", cls="muted"),
        Div(
            P(
                Strong("Dataset: "),
                "2015 Flight Delays and Cancellations from U.S. DOT (via Kaggle). ",
                "Contains ~5.8M domestic U.S. flights across 14 airlines and 600+ airports, ",
                "with on-time performance, delay, and cancellation data.",
            ),
            style="background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; margin: 1rem 0; font-size: 0.9rem;",
        ),
        Table(
            Thead(
                Tr(
                    Th("Column"),
                    Th("Type"),
                    Th("Description"),
                )
            ),
            Tbody(*table_rows),
            cls="schema-table",
        ),
        cls="schema-content",
    )

    return Div(
        modal_content,
        cls="schema-modal",
        hx_get="/",
        hx_target="body",
        hx_swap="outerHTML",
        hx_trigger="click from:body",
        onclick="if (event.target === this) { htmx.ajax('GET', '/', {target: 'body', swap: 'outerHTML'}); }",
    )


@rt
def index(req):
    hero = Div(
        P(
            "Explore U.S. flight delays from 2015 using natural language. ",
            "Ask about airlines, airports, specific dates, times of day, or delay patterns. ",
            "Try queries like 'worst delays by airline' or 'morning vs evening departure delays.' ",
            cls="muted lead",
        ),
        cls="hero",
    )

    schema_button = Div(
        Button(
            "Whats is this Dataset?",
            type="button",
            cls=f"{ButtonT.ghost}",
            hx_get=show_schema.to(),
            hx_target="#schema-modal",
            hx_swap="outerHTML",
            style="border-radius: 8px;",
        ),
        style="text-align: center; margin-top: 1rem; margin-bottom: 2rem;",
    )

    res_placeholder = Div(
        H3("Results"),
        Div(
            "Enter a natural language request above and click Run to see results.",
            cls="placeholder placeholder-idle muted",
        ),
        Div(
            Div(
                UkIcon("loader", height=16, cls="animate-spin mr-2"),
                Span("Loading results", cls="dots"),
                cls="flex items-center",
            ),
            cls="placeholder placeholder-loading muted",
        ),
        id="results",
        cls="panel",
    )

    # Hidden schema modal container
    schema_modal = Div(id="schema-modal")

    body = Div(
        hero, schema_button, mk_form(), res_placeholder, schema_modal, cls="space-y-8"
    )
    return mk_layout(body)


def result_card(*children: Any) -> Any:
    """Wrap results nicely in a card with consistent id for HTMX target."""
    # If no children provided, show a placeholder to ensure loading overlay is visible
    if not children:
        children = (
            Div(
                H3("Results"),
                P("Processing your query...", cls="muted"),
                cls="panel",
            ),
        )

    return Div(
        # Loading overlay that appears during HTMX requests
        Div(
            Div(
                UkIcon("loader", height=20, cls="animate-spin mr-2"),
                Span("Loading results", cls="dots"),
                cls="flex items-center text-lg",
            ),
            cls="loading-overlay htmx-indicator bg-surface-2 rounded-lg px-4 py-3 border border-border shadow-lg",
        ),
        *children,
        id="results",
        cls="relative",
    )


def render_eval_tile(
    title: str,
    ok: bool,
    description: str,
    details: Optional[str] = None,
    language: str = "text",
) -> Any:
    """Render a single eval tile.

    - Shows a compact pass/fail badge inline with the title.
    - Displays a clearer one-line description.
    - Renders any details or errors in a code-style block for readability.
    """
    badge = Span(
        "Pass" if ok else "Fail", cls=f"badge {'badge-pass' if ok else 'badge-fail'}"
    )
    blocks: list[Any] = [
        Div(H3(title), badge, cls="eval-title"),
        P(description, cls="eval-desc muted"),
    ]
    if not ok and details:
        # Prefer a code-style block for errors/details for clarity
        blocks.append(Div(Pre(Code(details)), cls="eval-code"))
    return Div(*blocks, cls="panel eval-tile")


def render_all_evals(sql_query: str) -> Any:
    """Construct all eval tiles for a given SQL string."""
    tiles: list[Any] = []
    results: list[bool] = []

    g_ok, g_msg = grammar_accepts(sql_query)
    results.append(g_ok)
    tiles.append(
        render_eval_tile(
            title="Grammar Checker",
            ok=g_ok,
            description="(Validates our Lark grammar for SQL)",
            details=g_msg if not g_ok else None,
        )
    )
    s_ok, s_msg = is_valid_clickhouse_sql(sql_query)
    results.append(s_ok)
    tiles.append(
        render_eval_tile(
            title="SQL Syntax Checker",
            ok=s_ok,
            description="(Checks if the SQL is valid)",
            details=s_msg if not s_ok else None,
        )
    )
    p_ok, p_problems = policy_check(sql_query)
    results.append(p_ok)
    p_detail = (
        None
        if p_ok
        else ("\n".join(f"- {x}" for x in p_problems) or "Unknown policy issue.")
    )
    tiles.append(
        render_eval_tile(
            title="Policy & Safeguards",
            ok=p_ok,
            description="(Checks for allowed columns and SQL operations)",
            details=p_detail,
        )
    )

    # Count failures and add class for vertical stacking if multiple failures
    failure_count = sum(1 for result in results if not result)
    grid_class = "eval-grid multiple-failures" if failure_count >= 2 else "eval-grid"

    return Div(H3("SQL Evals"), Div(*tiles, cls=grid_class), cls="evals-section")


@rt
async def run(nl_query: str = "", sess=None):
    try:
        if not nl_query.strip():
            return result_card(Alert("Please enter a query.", cls=AlertT.warning))

        sql_query, llm_text = await generate_sql_from_nl(nl_query.strip())

        # If no SQL was extracted, still show the LLM response and try evals
        if not sql_query:
            warn = Alert(
                "Could not extract SQL from model response.", cls=AlertT.warning
            )
            llm_block = (
                (Div(H3("Model response"), P(llm_text)))
                if llm_text
                else P("No model text returned.")
            )
            # Best‑effort: search for a SELECT in the text and run evals on it
            evals_section = None
            if isinstance(llm_text, str):
                m = re.search(r"SELECT\s", llm_text, flags=re.IGNORECASE)
                if m:
                    cand = llm_text[m.start() :].strip()
                    if not cand.endswith(";"):
                        cand += ";"
                    evals_section = render_all_evals(cand)
            parts = [warn, Divider(), llm_block]
            if evals_section is not None:
                parts += [Divider(), evals_section]
            return result_card(*parts)

        # Execute with ClickHouse
        if not key_id or not key_secret:
            raise RuntimeError("ClickHouse credentials are missing")

        try:
            csv_text = await asyncio.to_thread(
                run_clickhouse_query, sql_query, "CSVWithNames"
            )
        except requests.HTTPError as http_err:
            # Show both ClickHouse error and the LLM response + SQL
            err_alert = Alert(f"ClickHouse error: {http_err}", cls=AlertT.error)
            sql_block = CodeBlock(sql_query, language="sql")
            llm_block = (Div(H3("Model response"), P(llm_text))) if llm_text else None
            ch_detail = None
            try:
                if http_err.response is not None:
                    ch_detail = Pre(Code(http_err.response.text))
            except Exception:
                pass
            # Always include evals even when execution fails
            evals_section = render_all_evals(sql_query)
            parts = [
                err_alert,
                Divider(),
                DivLAligned(H3("Generated SQL")),
                sql_block,
                Divider(),
                evals_section,
            ]
            if llm_block is not None:
                parts += [Divider(), llm_block]
            if ch_detail is not None:
                parts += [Divider(), Div(H3("ClickHouse response")), ch_detail]
            return result_card(*parts)
        if csv_text.strip():
            df = await asyncio.to_thread(pd.read_csv, StringIO(csv_text))
        else:
            df = pd.DataFrame()

        # Let the LLM "see" the ClickHouse response and produce a brief summary
        llm_followup_text: Optional[str] = await llm_observe_clickhouse_result(
            nl_query, sql_query, csv_text
        )

        # Build polished two-column UI
        sql_pre = Pre(Code(sql_query), id="sql_text", cls="sql-view")
        llm_notes = llm_text if llm_text else None

        results_pane = Div(
            (
                render_df_table(df)
                if not df.empty
                else P("No rows returned.", cls="muted")
            ),
            cls="panel",
        )

        left_pane_children = []
        if llm_followup_text:
            left_pane_children.append(
                Div(
                    H3("Model's Summary of Results"),
                    Div(llm_followup_text, cls="marked"),
                    cls="panel",
                )
            )
        # Generated SQL panel (no copy buttons)
        left_pane_children.append(
            Div(H3("Generated SQL"), sql_pre, cls="panel sql-wrap")
        )

        grid = Div(
            Div(*left_pane_children, cls="left-stack"), results_pane, cls="results-grid"
        )
        # ---------- Evals on the generated SQL ----------
        evals_section = render_all_evals(sql_query)

        return result_card(
            Div(H3("Results"), Div(grid, evals_section), cls="results-section"),
            Script(
                "setTimeout(() => proc_htmx('.marked', e => e.innerHTML = marked.parse(e.textContent)), 100);"
            ),
        )

    except Exception as e:
        msg = str(e)
        if sess is not None:
            try:
                add_toast(sess, msg, "error")
            except Exception:
                pass
        err = Alert("Error: ", msg, cls=AlertT.error)
        # If we have model text in scope, surface it even on error
        try:
            extra = []
            if "llm_text" in locals() and llm_text:
                extra = [Divider(), Div(H3("Model response"), P(llm_text))]
            return result_card(err, *extra)
        except Exception:
            return result_card(err)


serve()
