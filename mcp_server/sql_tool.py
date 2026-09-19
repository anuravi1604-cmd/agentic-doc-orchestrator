"""
sql_tool.py
-----------
Safe, read-only SQL execution engine over SQLite tabular data with defensive
query verification and context-aware routing heuristics.

Safety guarantees:
1. Rejects any non-SELECT statements.
2. Defensively checks for forbidden DDL/DML keywords across the entire query string.
3. Rejects statement chaining (semicolons separating queries).
4. Opens SQLite connection in strict read-only URI mode (`?mode=ro`).
"""

from __future__ import annotations
import re
import sqlite3
from typing import Any, Dict, List, Tuple


_FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "create",
    "attach", "detach", "pragma", "replace", "truncate", "vacuum",
)

# Textual/policy keywords that strongly signal document retrieval intent
_DOC_INTENT_SIGNALS = {
    "policy", "policies", "runbook", "onboarding", "retained", "retention",
    "procedure", "procedures", "guideline", "guidelines", "reimbursement",
    "outage", "sev-1", "incident", "remote work", "work from home", "parental leave",
    "leave", "sla", "mfa", "authentication", "review", "reviews", "benefit",
    "benefits", "stipend", "referral", "bonus", "disaster recovery", "backup",
    "whistleblower", "vulnerability", "conference", "offboarding", "handbook",
    "standard", "standards", "specifies", "require", "requires",
}

# Structured domain terms relating to the tabular dataset
_STRUCTURED_METRICS = {"revenue", "headcount", "sales", "earnings", "income", "employees"}
_STRUCTURED_REGIONS = {"na", "eu", "apac", "latam", "north america", "europe"}
_STRUCTURED_TEMPORAL = {
    "q1", "q2", "q3", "q4", "quarter", "quarters", "quarterly", "2025", "2026",
    "q1-2025", "q2-2025", "q3-2025", "q4-2025", "q1-2026", "q2-2026", "q3-2026", "q4-2026",
}
_AGGREGATE_TERMS = {
    "total", "sum", "average", "avg", "highest", "lowest", "maximum", "minimum",
    "max", "min", "top", "bottom", "rank", "compare", "breakdown", "count",
}


class SafeSQLEngine:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def run_read_only(self, sql: str) -> Tuple[List[list], List[str]]:
        """Executes a single, verified read-only SELECT query against SQLite."""
        normalized = sql.strip().lower()

        if not normalized.startswith("select"):
            raise PermissionError("Security violation: Only SELECT statements are permitted.")

        for kw in _FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{kw}\b", normalized):
                raise PermissionError(f"Security violation: Forbidden keyword '{kw}' detected.")

        if ";" in sql.strip().rstrip(";"):
            raise PermissionError("Security violation: Chained/multiple statements are not permitted.")

        # Open in read-only URI mode to guarantee zero write capability at OS/driver level
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

    def classify_question(self, question: str) -> Dict[str, Any]:
        """
        Classifies whether a question targets structured relational tables or
        unstructured document retrieval using semantic intent and entity extraction.
        """
        q = question.lower()
        tokens = set(re.findall(r"\b[\w\-]+\b", q))

        has_doc_signal = any(sig in q for sig in _DOC_INTENT_SIGNALS)
        has_metric = bool(tokens & _STRUCTURED_METRICS) or any(m in q for m in ("revenue", "headcount"))
        has_region = bool(tokens & _STRUCTURED_REGIONS)
        has_temporal = bool(tokens & _STRUCTURED_TEMPORAL)
        has_aggregate = bool(tokens & _AGGREGATE_TERMS) or any(p in q for p in ("how many quarters", "number of quarters"))
        has_db_keyword = any(k in q for k in ("database", "table", "sql", "rows", "records recorded"))

        # Question asks for structured metrics or aggregations over the tabular schema
        is_structured = False
        reason = ""

        if has_doc_signal and not has_metric:
            # e.g., "How many days can I work remotely?" or "Can we compare our security policies?"
            is_structured = False
            reason = "Question targets company policies, procedures, or operational documentation."
        elif has_metric or (has_aggregate and (has_region or has_temporal)) or (has_db_keyword and (has_temporal or has_region)):
            # e.g., "What was total revenue in Q1?", "Which region had highest headcount?", "How many quarters are recorded in the database?"
            is_structured = True
            reason = "Question targets tabular numerical data (revenue, headcount, quarters, regions, database records)."
        elif ("make in" in q or "generated in" in q) and (has_region or has_temporal):
            # e.g., "What did the company make in NA during Q2?"
            is_structured = True
            reason = "Question queries financial performance across region/quarter."
        else:
            is_structured = False
            reason = "Question is descriptive or conceptual; routing to document retrieval."

        return {
            "route": "sql_query" if is_structured else "semantic_search",
            "is_structured": is_structured,
            "reason": reason,
        }

    def looks_structured(self, question: str) -> bool:
        return self.classify_question(question)["is_structured"]
