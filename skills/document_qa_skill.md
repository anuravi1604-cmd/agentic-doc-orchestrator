---
name: Document Intelligence Q&A
version: 1.0.0
description: Operational guidelines for orchestrating questions across unstructured documents and structured tabular data.
tools:
  - classify_document
  - semantic_search
  - sql_query
---

# Skill: Document Intelligence Q&A

## When to use this skill
Use this skill whenever a user asks a question that requires looking up information
in either unstructured company documents (policies, runbooks, onboarding docs,
engineering guidelines) or structured tabular data (quarterly revenue, regional figures,
headcount metrics) exposed through the `doc-intelligence-mcp` MCP server.

## Available tools (via MCP)
- `classify_document`: Classifies whether the query targets structured tabular metrics (revenue, headcount) or unstructured documents (policies, processes).
- `semantic_search`: Hybrid lexical and sub-word retrieval over unstructured documentation with Reciprocal Rank Fusion.
- `sql_query`: Executes a safe, read-only SELECT statement against the `quarterly_revenue` database.

## Procedure
1. Call `classify_document` with the user's raw question to determine whether to route to `sql_query` or `semantic_search`.
2. If routed to `sql_query`: Inspect the `quarterly_revenue` schema, formulate a single read-only SELECT query covering relevant filters (quarter, region) and aggregations (sum, average, max), and call `sql_query`.
3. If routed to `semantic_search`: Call `semantic_search` with the user query and `top_k=3` to retrieve relevant document passages and metadata.
4. Synthesize a grounded, concise answer citing the retrieved document ID or formatting the tabular values.
5. If a tool call fails or returns an error, explain the limitation clearly rather than attempting unsafe workarounds.

## Notes
- Structured numeric figures must always come from `sql_query` (the source of truth), never estimated from text retrieval.
- Policy and procedural queries must always go to `semantic_search`, even if containing words like "compare" or "how many".
- Never execute write, DDL, or chained SQL queries.
