import os
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from fasthtml.common import *
from typing import Optional, Tuple, Any
from monsterui.all import *
from grammar import clickhouse_flights_grammar, sql_lark_cfg_tool
from openai import AsyncAzureOpenAI

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
        "You translate natural language into valid ClickHouse SQL over table flights_df. "
        "Use only constructs allowed by the provided grammar. Always end with a semicolon. "
        "User request: " + natural_language_query
    )

    resp = await client.responses.create(
        model="gpt-5-mini",
        input=prompt,
        text={"format": {"type": "text"}},
        tools=[sql_lark_cfg_tool],
        parallel_tool_calls=False,
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
    return Div(NotStr(table_html))


# ----- FastHTML app -----
hdrs = Theme.blue.headers(highlightjs=True)
app, rt = fast_app(hdrs=hdrs)
setup_toasts(app)


def mk_layout(content: Any) -> Any:
    """Standard page layout wrapper."""
    title = "Flights CFG Demo"
    subtitle = "Query ClickHouse with natural language (LLM -> SQL)"
    header = DivLAligned(H1(title), Subtitle(subtitle))
    return Titled(title, Container(header, content))


def mk_form() -> Any:
    """Create the query form with HTMX attributes and indicator."""
    example = "sum the total of all flights in the last 30 hours of today in 2015"
    input_box = Textarea(
        id="nl_query", name="nl_query", rows=3, placeholder=f"e.g. {example}"
    )
    spinner = Loading(cls="htmx-indicator", htmx_indicator=True)
    form = Form(
        hx_post=run.to(),
        hx_target="#results",
        hx_swap="outerHTML",
        hx_indicator="#loading",
    )(
        Grid(
            input_box,
            Button("Run", type="submit", cls=ButtonT.primary),
        ),
        Div(spinner, id="loading"),
    )
    return form


@rt
def index(req):
    hero = Card(
        P(
            "Ask a question about flights. We'll convert it to SQL and run it in ClickHouse."
        ),
        footer=Div(P("Table: flights_df")),
    )

    res_placeholder = Card(P("Results will appear here."), id="results")

    body = Div(hero, mk_form(), res_placeholder, cls="space-y-4")
    return mk_layout(body)


def result_card(*children: Any) -> Any:
    """Wrap results nicely in a card with consistent id for HTMX target."""
    return Card(*children, id="results")


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
            csv_text = run_clickhouse_query(sql_query, format_type="CSVWithNames")
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
        df = pd.read_csv(StringIO(csv_text)) if csv_text.strip() else pd.DataFrame()

        # Build UI
        sql_block = CodeBlock(sql_query, language="sql")
        llm_block = (Div(H3("Model notes"), P(llm_text))) if llm_text else None

        if df.empty:
            tbl = P("No rows returned.")
        else:
            tbl = render_df_table(df)

        header = DivLAligned(H3("Generated SQL"))
        parts = [header, sql_block]
        if llm_block is not None:
            parts.append(Divider())
            parts.append(llm_block)
        parts.append(Divider())
        parts.append(Div(H3("Results")))
        parts.append(tbl)
        return result_card(*parts)

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
