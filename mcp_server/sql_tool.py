"""
sql_tool.py
-----------
A safe, read-only SQL execution layer over a sample SQLite database, plus a
lightweight classifier that decides whether a natural-language question
should be routed to SQL (structured/numeric) or semantic search
(unstructured/descriptive) -- mirroring ContextIQ's CSV/Excel vs. document
auto-routing.

Safety measures (important for anything exposed to an LLM-driven agent):
    - Rejects any statement that isn't a single SELECT.
    - Blocks write/DDL keywords defensively even inside a SELECT (e.g. a
      SELECT wrapped around a subquery containing INSERT is still rejected
      because we check the whole normalized string, not just the first token).
    - Opens the connection in a way that only ever reads.
"""

from __future__ import annotations
import re
import sqlite3
from typing import List, Tuple


_FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "create",
    "attach", "detach", "pragma", "replace", "truncate", "vacuum",
)

_STRUCTURED_HINTS = (
    "how many", "total", "average", "avg", "sum", "count", "maximum",
    "minimum", "highest", "lowest", "compare", "%", "greater than",
    "less than", "top ", "revenue", "score", "number of",
)


class SafeSQLEngine:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def run_read_only(self, sql: str) -> Tuple[List[list], List[str]]:
        normalized = sql.strip().lower()

        if not normalized.startswith("select"):
            raise PermissionError("Only SELECT statements are permitted.")

        for kw in _FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{kw}\b", normalized):
                raise PermissionError(f"Statement contains forbidden keyword: '{kw}'")

        if ";" in sql.strip().rstrip(";"):
            raise PermissionError("Multiple statements are not permitted.")

        # Open strictly read-only via URI mode so even a bug above can't write.
        uri = f"file:{self.db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            columns = [d[0] for d in cur.description] if cur.description else []
            return [list(r) for r in rows], columns
        finally:
            conn.close()

    def looks_structured(self, question: str) -> bool:
        """Cheap keyword-based router; a real system would use a trained
        classifier or an LLM call, but this keeps the demo fast and offline."""
        q = question.lower()
        return any(hint in q for hint in _STRUCTURED_HINTS)
