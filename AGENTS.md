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
