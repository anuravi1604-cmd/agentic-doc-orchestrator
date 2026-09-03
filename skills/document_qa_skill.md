# Skill: Document Intelligence Q&A

## When to use this skill
Use this skill whenever a user asks a question that requires looking something
up in either unstructured company documents (policies, runbooks, onboarding
docs) or structured tabular data (quarterly revenue, headcount) exposed
through the `doc-intelligence-mcp` MCP server.

## Available tools (via MCP)
- `classify_document(question)` -> tells you whether to use `sql_query` or
  `semantic_search` for a given question. Always call this first unless the
  user has already told you which data source to use.
- `semantic_search(query, top_k)` -> hybrid lexical+semantic search over
  unstructured text documents. Use for descriptive/policy questions.
- `sql_query(sql)` -> read-only SQL execution over the `quarterly_revenue`
  table `(quarter, region, revenue, headcount)`. Use for numeric/aggregate
  questions. Only ever write a single SELECT statement -- no writes, no DDL,
  no multiple statements.

## Procedure
1. Call `classify_document` with the user's raw question.
2. If routed to `sql_query`: draft a single, minimal SELECT statement against
   `quarterly_revenue` that answers the question, then call `sql_query`.
3. If routed to `semantic_search`: call it with the user's question as the
   query and `top_k=3`.
4. Synthesize a direct, concise answer from the tool output. Do not restate
   the raw rows/passages verbatim -- summarize what they mean for the
   question asked.
5. If a tool call errors (e.g. `sql_query` rejects a write statement), do not
   retry with a workaround that bypasses the restriction -- explain the
   limitation to the user instead.

## Notes
- This skill assumes the `doc-intelligence-mcp` server (see `mcp_server/`)
  is already running and reachable by the calling agent.
- Numeric questions should always go through `sql_query`, never be estimated
  from `semantic_search` results, since the structured table is the source
  of truth for those figures.
