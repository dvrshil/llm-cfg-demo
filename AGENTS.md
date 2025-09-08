# Repository Guidelines

## Project Structure & Module Organization
- `main.py` — App entrypoint (FastHTML server, routes, UI, Azure OpenAI calls, ClickHouse query, eval tiles).
- `grammar.py` — Lark CFG and the grammar tool used to constrain SQL generation.
- `evals.py` — Minimal eval helpers (CFG acceptance, ClickHouse parse, policy checks) and a set of example natural‑language prompts used for "Suggested inputs" in the UI.
- `README.md` — User‑level overview and how to interact with the demo.
- `requirements.txt` — Python dependencies.

## Build, Test, and Development Commands
- Install: `pip install -r requirements.txt`
- Run app (dev): `python main.py` (serves on http://localhost:5001)
- Lint (optional): `python -m pyflakes main.py`
- Grammar sanity: import `grammar.py` and parse a few SQL examples with Lark. The UI evals run automatically after each query; if execution fails, evals still render against the generated SQL to aid debugging.

## Coding Style & Naming Conventions
- Python 3.10+; 4‑space indentation; prefer clear, descriptive names.
- Keep functions small and single‑purpose; avoid unnecessary globals.
- UI copy: concise, plain language; avoid exposing internal column names unless needed.
- CSS lives alongside the app code in `main.py`; extend the existing style block rather than adding frameworks. Chips/suggestions styles are in the same block.

## Testing Guidelines
- Functional path: in the UI, submit prompts and verify (1) Generated SQL, (2) Results table, (3) “SQL Evals” tiles all look correct.
- Parser checks: `sqlglot.parse_one(sql, read="clickhouse")` should succeed for valid queries.
- CFG checks: Lark should accept grammar‑conformant SQL and reject out‑of‑grammar strings.

## Commit & Pull Request Guidelines
- Commits: small, scoped, imperative (e.g., "Tighten grammar", "Polish SQL panel").
- PRs: include a brief rationale, screenshots for UI changes, and a note if grammar/policy lists changed.
- Link related issues; call out any env var or dependency changes.

## Security & Configuration Tips
- Secrets come from `.env` (ClickHouse and Azure OpenAI). Never commit real keys; rotate if exposed.
- The CFG locks queries to `SELECT` over `flights_df`. If you expand grammar, mirror constraints in `evals.py` policy checks to preserve safety.

# Core Project Spec where we started, and wanna achieve + overshoot in a nice way:
```
# Context Free Grammars + Eval Toy

https://cookbook.openai.com/examples/gpt-5/gpt-5_new_params_and_tools#3-contextfree-grammar-cfg

<aside>
✨

GPT-5 has newly added support for Context Free Grammars (CFG). This allows you to constrain the model’s output in a way in a much more powerful way. You’ll create a small app for experimenting with CFG’s + a few evals for proving it works!

</aside>

### Timeline:

Estimated Commitment: 6.5 hours

Requested Delivery: 3 days

### Deliverables

*Please send deliverables within 3 days of starting project! If you have scheduling conflicts that prevent this (or disagree with our timeline!), just let us know before starting.*

- A deployed app where someone can type in a natural language query “sum the total of all orders placed in the last 30 hours” and see data from clickhouse returned.
    - You don’t need to render graphs/tables/etc. Raw JSON response is fine!
    - You must use GPT-5’s newly added Context Free Grammar
- 3+ Evals for the generation of the CFG. You can roll your own eval framework, or use anything off the shelf. Don’t overthink this!
    - The evals can be run from the app you deploy, a script, etc.
- GitHub + source code!

### Getting Started

- Use [Tinybird](http://tinybird.com) or [ClickHouse Cloud](http://clickhouse.cloud) to ingest some CSV data. Choose any large, 1000+ row dataset you’d like.
- Define a CFG for the ClickHouse table.
- The app you make should have a prompt for typing in any query, and seeing the response that is returned from the API.
- Make sure to also make just a few (3 is okay!) evals.
```
