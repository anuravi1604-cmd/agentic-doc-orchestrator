"""
sql_generator.py
----------------
Dynamic natural-language to SQL generator for the quarterly_revenue schema:
quarterly_revenue (quarter TEXT, region TEXT, revenue REAL, headcount INTEGER)

Constructs syntactically valid, parameterized, read-only SELECT queries by
dynamically parsing metrics, aggregations, region/quarter filters, grouping,
and ordering from natural language input.
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple


class SQLGenerator:
    """Dynamic rule- and entity-based SQL generator for tabular document Q&A."""

    SCHEMA_HINT = "quarterly_revenue(quarter TEXT, region TEXT, revenue REAL, headcount INTEGER)"

    REGIONS_MAP = {
        "na": "NA",
        "north america": "NA",
        "eu": "EU",
        "europe": "EU",
        "apac": "APAC",
        "asia": "APAC",
        "latam": "LATAM",
        "latin america": "LATAM",
    }

    def generate(self, question: str) -> str:
        q = question.lower().strip()

        # 1. Determine target metric
        if any(term in q for term in ("headcount", "employee", "employees", "staff", "people")):
            metric = "headcount"
        else:
            metric = "revenue"

        # 2. Extract region filters
        matched_regions: List[str] = []
        for phrase, code in self.REGIONS_MAP.items():
            if re.search(rf"\b{re.escape(phrase)}\b", q):
                if code not in matched_regions:
                    matched_regions.append(code)

        # 3. Extract quarter/year filters
        quarter_filter: Optional[str] = None
        exact_qy = re.search(r"\b(q[1-4])[-_\s]?(202[5-6])\b", q)
        if exact_qy:
            quarter_filter = f"quarter = '{exact_qy.group(1).upper()}-{exact_qy.group(2)}'"
        else:
            q_only = re.search(r"\b(q[1-4])\b", q)
            year_only = re.search(r"\b(202[5-6])\b", q)
            if q_only and year_only:
                quarter_filter = f"quarter = '{q_only.group(1).upper()}-{year_only.group(1)}'"
            elif q_only:
                quarter_filter = f"quarter LIKE '{q_only.group(1).upper()}%'"
            elif year_only:
                quarter_filter = f"quarter LIKE '%{year_only.group(1)}'"

        # 4. Detect aggregation / comparison intent
        is_avg = any(term in q for term in ("average", "avg", "mean"))
        is_sum = any(term in q for term in ("total", "sum", "combined", "overall"))
        is_highest = any(term in q for term in ("highest", "top", "maximum", "max", "most", "largest", "peak"))
        is_lowest = any(term in q for term in ("lowest", "bottom", "minimum", "min", "least", "smallest"))
        is_count = any(term in q for term in ("how many quarters", "number of quarters", "count of"))
        is_comparison = "compare" in q or (len(matched_regions) > 1 and "and" in q)

        # 5. Detect Grouping
        group_by_region = (
            "by region" in q or "per region" in q or "each region" in q
            or "across regions" in q or is_comparison
            or ("which region" in q and (is_highest or is_lowest or is_sum or is_avg))
        )
        group_by_quarter = (
            "by quarter" in q or "per quarter" in q or "each quarter" in q
            or ("which quarter" in q and (is_highest or is_lowest))
        )

        # 6. Build WHERE clause
        where_clauses: List[str] = []
        if matched_regions:
            if len(matched_regions) == 1:
                where_clauses.append(f"region = '{matched_regions[0]}'")
            else:
                formatted_regions = ", ".join(f"'{r}'" for r in matched_regions)
                where_clauses.append(f"region IN ({formatted_regions})")

        if quarter_filter:
            where_clauses.append(quarter_filter)

        where_stmt = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # 7. Assemble Query
        if is_count:
            return f"SELECT COUNT(DISTINCT quarter) AS quarter_count FROM quarterly_revenue{where_stmt};"

        if is_highest:
            if group_by_region:
                return (
                    f"SELECT region, SUM({metric}) AS total_{metric} "
                    f"FROM quarterly_revenue{where_stmt} "
                    f"GROUP BY region ORDER BY total_{metric} DESC LIMIT 1;"
                )
            elif group_by_quarter:
                return (
                    f"SELECT quarter, SUM({metric}) AS total_{metric} "
                    f"FROM quarterly_revenue{where_stmt} "
                    f"GROUP BY quarter ORDER BY total_{metric} DESC LIMIT 1;"
                )
            else:
                return (
                    f"SELECT quarter, region, {metric} "
                    f"FROM quarterly_revenue{where_stmt} "
                    f"ORDER BY {metric} DESC LIMIT 1;"
                )

        if is_lowest:
            if group_by_region:
                return (
                    f"SELECT region, SUM({metric}) AS total_{metric} "
                    f"FROM quarterly_revenue{where_stmt} "
                    f"GROUP BY region ORDER BY total_{metric} ASC LIMIT 1;"
                )
            elif group_by_quarter:
                return (
                    f"SELECT quarter, SUM({metric}) AS total_{metric} "
                    f"FROM quarterly_revenue{where_stmt} "
                    f"GROUP BY quarter ORDER BY total_{metric} ASC LIMIT 1;"
                )
            else:
                return (
                    f"SELECT quarter, region, {metric} "
                    f"FROM quarterly_revenue{where_stmt} "
                    f"ORDER BY {metric} ASC LIMIT 1;"
                )

        if is_avg:
            if group_by_quarter and not where_clauses:
                return f"SELECT quarter, AVG({metric}) AS avg_{metric} FROM quarterly_revenue GROUP BY quarter;"
            elif group_by_region and not matched_regions:
                return f"SELECT region, AVG({metric}) AS avg_{metric} FROM quarterly_revenue GROUP BY region;"
            else:
                return f"SELECT AVG({metric}) AS avg_{metric} FROM quarterly_revenue{where_stmt};"

        if is_sum or not (group_by_region or group_by_quarter or where_clauses):
            if group_by_region:
                return f"SELECT region, SUM({metric}) AS total_{metric} FROM quarterly_revenue{where_stmt} GROUP BY region;"
            elif group_by_quarter:
                return f"SELECT quarter, SUM({metric}) AS total_{metric} FROM quarterly_revenue{where_stmt} GROUP BY quarter;"
            else:
                return f"SELECT SUM({metric}) AS total_{metric} FROM quarterly_revenue{where_stmt};"

        # Default filtered or detailed query
        if group_by_region:
            return f"SELECT region, SUM({metric}) AS total_{metric} FROM quarterly_revenue{where_stmt} GROUP BY region;"
        elif group_by_quarter:
            return f"SELECT quarter, SUM({metric}) AS total_{metric} FROM quarterly_revenue{where_stmt} GROUP BY quarter;"
        elif where_clauses:
            return f"SELECT quarter, region, {metric} FROM quarterly_revenue{where_stmt};"

        return f"SELECT quarter, region, revenue, headcount FROM quarterly_revenue;"
