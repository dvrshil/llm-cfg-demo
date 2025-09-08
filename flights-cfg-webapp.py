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
                    sql_query = candidate
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
                cand = llm_text[m.start() :].strip()
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
            "summary or 2-3 observations. If the table is empty, state that clearly."
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
_theme_hdrs = Theme.blue.headers(highlightjs=True)
# App-specific minimal CSS to polish layout
_app_css = Style(
    """
    :root { --surface: #0c0f13; --surface-2:#0f141a; --border:#222933; --shadow: 0 10px 24px rgba(0,0,0,.35); }
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
    .results-grid { display:grid; grid-template-columns: 1fr; gap: 1.25rem; align-items:start; margin-top: 4rem; }
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
    #results table { min-width: 100%; width: max-content; border-collapse: separate; border-spacing: 0; white-space: nowrap; }
    #results thead th { position: sticky; top: 0; background: var(--surface-2); z-index: 1; }
    #results th, #results td { padding: .5rem .75rem; border-bottom: 1px solid var(--border); border-right: 1px solid var(--border); font-size: .92rem; }
    #results th:first-child, #results td:first-child { border-left: 1px solid var(--border); }
    #results tr:hover td { background: rgba(255,255,255,.02); }
    .tight { margin-block: .5rem; }
    /* Placeholder phases inside results */
    #results .placeholder { display:block; margin: .25rem 0; }
    #results .placeholder-loading { display:none; }
    #results.htmx-request .placeholder-idle { display:none; }
    #results.htmx-request .placeholder-loading { display:block; }
    #results .dots:after { content: '…'; animation: dots 1.2s steps(4,end) infinite; }
    @keyframes dots { 0%,20%{content:''} 40%{content:'.'} 60%{content:'..'} 80%,100%{content:'...'} }
    /* Evals styling */
    .evals-section { margin-top: 2.5rem; }
    .evals-section > h3 { margin: 0 0 1rem 0; }
    .eval-grid { display:grid; grid-template-columns: 1fr; gap: 2rem; }
    @media (min-width: 900px) { .eval-grid { grid-template-columns: repeat(3, 1fr); } }
    .eval-tile { position: relative; }
    .status-pass { color: #32d296; font-weight: 600; }
    .status-fail { color: #ff6b6b; font-weight: 600; }
    .eval-title { display:flex; align-items:center; justify-content: space-between; gap: .75rem; margin-bottom: .35rem; }
    .badge { border:1px solid var(--border); border-radius: 6px; padding: .15rem .5rem; font-size: .78rem; font-weight: 650; line-height: 1.2; }
    .badge-pass { color:#32d296; background: rgba(50,210,150,.08); border-color: rgba(50,210,150,.35); }
    .badge-fail { color:#ff6b6b; background: rgba(255,107,107,.10); border-color: rgba(255,107,107,.40); }
    .eval-desc { margin: .15rem 0 .5rem 0; opacity: .9; }
    .eval-code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                font-size: .85rem; line-height: 1.3; white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere;
                background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: .6rem .75rem;
                max-height: 200px; width: 100%; overflow: auto; margin: .25rem 0; }
    """
)
hdrs = (*_theme_hdrs, _app_css)
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

    body = Div(hero, mk_form(), res_placeholder, cls="space-y-8")
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


@rt
async def run(nl_query: str = "", sess=None):
    try:
        if not nl_query.strip():
            return result_card(Alert("Please enter a query.", cls=AlertT.warning))

        sql_query, llm_text = await generate_sql_from_nl(nl_query.strip())

        # If no SQL was extracted, still show the LLM response so users can see what happened
        if not sql_query:
            warn = Alert(
                "Could not extract SQL from model response.", cls=AlertT.warning
            )
            llm_block = (
                (Div(H3("Model response"), P(llm_text)))
                if llm_text
                else P("No model text returned.")
            )
            return result_card(warn, Divider(), llm_block)

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
            parts = [err_alert, Divider(), DivLAligned(H3("Generated SQL")), sql_block]
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
            H3("Results"),
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
                Div(H3("Model on Results"), P(llm_followup_text), cls="panel")
            )
        if llm_notes:
            left_pane_children.append(Div(H3("Model Notes"), P(llm_notes), cls="panel"))
        # Generated SQL panel (no copy buttons)
        left_pane_children.append(
            Div(H3("Generated SQL"), sql_pre, cls="panel sql-wrap")
        )

        grid = Div(
            Div(*left_pane_children, cls="left-stack"), results_pane, cls="results-grid"
        )
        # ---------- Evals on the generated SQL ----------
        eval_tiles: list[Any] = []

        # 1) Grammar acceptance
        g_ok, g_msg = grammar_accepts(sql_query)
        eval_tiles.append(
            render_eval_tile(
                title="Grammar Acceptance",
                ok=g_ok,
                description="Checks the SQL against the flights CFG.",
                details=g_msg if not g_ok else None,
                language="text",
            )
        )

        # 2) General SQL syntax under ClickHouse
        s_ok, s_msg = is_valid_clickhouse_sql(sql_query)
        eval_tiles.append(
            render_eval_tile(
                title="ClickHouse Parse",
                ok=s_ok,
                description="Parses the SQL with the ClickHouse dialect (sqlglot).",
                details=s_msg if not s_ok else None,
                language="text",
            )
        )

        # 3) Policy / safety
        p_ok, p_problems = policy_check(sql_query)
        p_desc = "Read-only SELECT; only whitelisted columns/functions; no joins, unions, CTEs, subqueries, or window clauses."
        # Format problems as a simple markdown-style bullet list in a code view
        p_detail = None
        if not p_ok:
            if p_problems:
                p_detail = "\n".join(f"- {x}" for x in p_problems)
            else:
                p_detail = "Unknown policy issue."
        eval_tiles.append(
            render_eval_tile(
                title="Policy & Safeguards",
                ok=p_ok,
                description=p_desc,
                details=p_detail,
                language="text",
            )
        )

        evals_section = Div(
            H3("SQL Evals"), Div(*eval_tiles, cls="eval-grid"), cls="evals-section"
        )

        return result_card(Div(grid, evals_section))

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
