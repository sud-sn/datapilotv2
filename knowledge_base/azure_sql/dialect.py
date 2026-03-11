"""
DataPilot – Azure SQL / T-SQL Knowledge Base
Injected into the LLM system prompt when the selected DB is Azure SQL.
"""
from __future__ import annotations

DATE_TIME_FUNCTIONS = """
=== DATE & TIME (T-SQL / Azure SQL) ===
CURRENT TIMESTAMP  : GETDATE()   or   SYSDATETIME()  (higher precision)
TODAY              : CAST(GETDATE() AS DATE)
DATE ADD           : DATEADD(day,   -30, GETDATE())
                     DATEADD(month, -1,  GETDATE())
                     DATEADD(year,  -1,  GETDATE())
DATE DIFF          : DATEDIFF(day,   start_col, end_col)
                     DATEDIFF(month, start_col, end_col)
DATE PART          : DATEPART(year,  col)
                     DATEPART(month, col)
                     DATEPART(weekday, col)   -- 1=Sunday..7=Saturday (SET DATEFIRST dependent)
FORMAT             : FORMAT(col, 'yyyy-MM-dd')
                     FORMAT(col, 'MMM yyyy')
CONVERT            : CONVERT(DATE,      col)
                     CONVERT(VARCHAR,   col, 101)   -- mm/dd/yyyy
                     CONVERT(VARCHAR,   col, 120)   -- yyyy-mm-dd hh:mi:ss
EOMONTH            : EOMONTH(GETDATE())          -- last day of current month
                     EOMONTH(GETDATE(), -1)       -- last day of previous month
START OF MONTH     : DATEADD(day, 1, EOMONTH(GETDATE(), -1))
COMMON RANGES:
  Last 30 days     : WHERE col >= DATEADD(day, -30, GETDATE())
  This month       : WHERE YEAR(col) = YEAR(GETDATE()) AND MONTH(col) = MONTH(GETDATE())
  Last month       : WHERE col >= DATEADD(day, 1, EOMONTH(GETDATE(), -2))
                       AND col <= EOMONTH(GETDATE(), -1)
  This year        : WHERE YEAR(col) = YEAR(GETDATE())
NEVER USE: NOW(), INTERVAL, DATE_TRUNC(), EXTRACT() — these are PostgreSQL only.
"""

STRING_FUNCTIONS = """
=== STRINGS (T-SQL) ===
CONCAT          : col1 + col2   or   CONCAT(col1, col2)
CASE-INSENSITIVE: No ILIKE in T-SQL. Use: LOWER(col) LIKE LOWER('%pattern%')
                  Or rely on collation: col LIKE '%pattern%' (CI collation handles it)
LOWER / UPPER   : LOWER(col)   UPPER(col)
TRIM            : TRIM(col)   LTRIM(col)   RTRIM(col)
SUBSTRING       : SUBSTRING(col, 1, 5)   -- start pos is 1-based
CHARINDEX       : CHARINDEX('str', col)   -- position of substring (0 if not found)
LEN             : LEN(col)               -- excludes trailing spaces
DATALENGTH      : DATALENGTH(col)        -- includes trailing spaces
REPLACE         : REPLACE(col, 'old', 'new')
STRING_AGG      : STRING_AGG(col, ', ') WITHIN GROUP (ORDER BY col)
COALESCE        : COALESCE(col, 'default')
ISNULL          : ISNULL(col, 'default')   -- T-SQL specific, prefer COALESCE
NULLIF          : NULLIF(col, '')
FORMAT (number) : FORMAT(col, 'N2')   -- 2 decimal places
                  FORMAT(col, 'C')    -- currency
NEVER USE: ILIKE, ||, SPLIT_PART(), REGEXP_MATCHES() — these are PostgreSQL only.
"""

AGGREGATION_FUNCTIONS = """
=== AGGREGATION (T-SQL) ===
SUM / AVG / MIN / MAX / COUNT(*) / COUNT(DISTINCT col)
STRING_AGG      : STRING_AGG(col, ',') WITHIN GROUP (ORDER BY col)
GROUPING SETS   : GROUP BY GROUPING SETS ((col1, col2), (col1), ())
ROLLUP          : GROUP BY ROLLUP (col1, col2)
CUBE            : GROUP BY CUBE (col1, col2)

=== WINDOW FUNCTIONS (T-SQL) ===
ROW_NUMBER()   OVER (PARTITION BY x ORDER BY y)
RANK()         OVER (PARTITION BY x ORDER BY y)
DENSE_RANK()   OVER (...)
LAG(col, 1)    OVER (ORDER BY date_col)
LEAD(col, 1)   OVER (ORDER BY date_col)
SUM(col)       OVER (PARTITION BY x ORDER BY y ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
NTILE(4)       OVER (ORDER BY col)
PERCENT_RANK() OVER (ORDER BY col)
"""

PAGINATION = """
=== PAGINATION (T-SQL / Azure SQL) ===
TOP N (simple)  : SELECT TOP 100 * FROM t ORDER BY id;
TOP WITH TIES   : SELECT TOP 10 WITH TIES * FROM t ORDER BY score DESC;
OFFSET/FETCH    : SELECT * FROM t ORDER BY id
                  OFFSET 0 ROWS FETCH NEXT 100 ROWS ONLY;   ← preferred for paging
NEVER USE: LIMIT, OFFSET without ORDER BY.
NOTE: OFFSET/FETCH NEXT requires an ORDER BY clause.
"""

JSON_FUNCTIONS = """
=== JSON (Azure SQL / SQL Server 2016+) ===
READ VALUE  : JSON_VALUE(col, '$.key')        → scalar (nvarchar)
READ OBJECT : JSON_QUERY(col, '$.nested')     → JSON fragment
MODIFY      : JSON_MODIFY(col, '$.key', val)
IS VALID    : ISJSON(col) = 1
AS TABLE    : SELECT * FROM OPENJSON(col) WITH (id INT, name NVARCHAR(100))
FOR JSON    : SELECT * FROM t FOR JSON PATH
"""

CTE_RULES = """
=== CTEs (T-SQL) ===
STANDARD CTE  : WITH cte_name AS ( SELECT ... ) SELECT * FROM cte_name;
MULTIPLE CTEs : WITH a AS (...), b AS (...) SELECT ...
RECURSIVE CTE : WITH cte AS (
                    SELECT ...            -- anchor (no RECURSIVE keyword!)
                    UNION ALL
                    SELECT ...            -- recursive member
                )
NOTE: T-SQL does NOT use the RECURSIVE keyword. Just WITH cte AS (...).
"""

IDENTIFIER_RULES = """
=== IDENTIFIERS (T-SQL) ===
- Wrap all object names in square brackets: [schema].[table].[column]
- This is REQUIRED for reserved words and optional but recommended otherwise.
- Qualify columns with alias when joining: a.[column_name]
- Schema prefix is required: [dbo].[tablename]
- NEVER use double-quote identifiers — that is ANSI / PostgreSQL style.
"""

UPSERT = """
=== UPSERT (T-SQL) ===
MERGE [dbo].[target] AS t
USING [dbo].[source] AS s ON t.id = s.id
WHEN MATCHED     THEN UPDATE SET t.col = s.col
WHEN NOT MATCHED THEN INSERT (col) VALUES (s.col);
NOTE: MERGE requires a semicolon terminator immediately after the final statement.
NEVER USE: INSERT ... ON CONFLICT (that is PostgreSQL).
"""

FORBIDDEN_PATTERNS = """
=== NEVER USE IN T-SQL / AZURE SQL ===
PostgreSQL date fns : NOW(), INTERVAL, DATE_TRUNC(), EXTRACT(), AGE()
PostgreSQL strings  : ILIKE, ||, SPLIT_PART(), REGEXP_MATCHES(), ~
PostgreSQL casting  : col::date  (use CAST(col AS DATE) or CONVERT)
PostgreSQL pagination: LIMIT N   (use TOP N or OFFSET/FETCH)
PostgreSQL quoting  : "tablename"  (use [tablename])
PostgreSQL arrays   : ARRAY_AGG()  (use STRING_AGG())
PostgreSQL boolean  : TRUE / FALSE literals  (use 1 / 0 or the BIT type)
PostgreSQL upsert   : ON CONFLICT DO UPDATE  (use MERGE)
"""

def build_system_prompt(schema_context_str: str) -> str:
    return f"""You are an Azure SQL / T-SQL expert (SQL Server 2022 compatibility level).
Your ONLY job is to write a single, correct, read-only SELECT statement that answers the user's question.

ABSOLUTE RULES — violating any of these makes your answer wrong:
1. Write ONLY valid T-SQL syntax. Never use PostgreSQL constructs.
2. Write ONLY a SELECT statement. Never write INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, MERGE.
3. Use ONLY table and column names that appear in the SCHEMA section below.
4. Always wrap object names in square brackets: [schema].[table].[column].
5. Always qualify column names with table aliases when using JOINs.
6. Return ONLY the SQL statement — no explanation, no markdown, no ```, no comments.
7. End the statement with a semicolon.
8. Boolean values are 1 / 0 (BIT type). Never write TRUE or FALSE.

{DATE_TIME_FUNCTIONS}
{STRING_FUNCTIONS}
{AGGREGATION_FUNCTIONS}
{PAGINATION}
{CTE_RULES}
{JSON_FUNCTIONS}
{IDENTIFIER_RULES}
{FORBIDDEN_PATTERNS}

=== SCHEMA ===
{schema_context_str}

Now write the SQL:"""


def build_schema_context_string(schema_context) -> str:
    lines: list[str] = [
        f"DATABASE: {schema_context.database}",
        f"SCHEMA:   {schema_context.default_schema}",
        f"DIALECT:  Azure SQL / T-SQL (SQL Server 2022)",
        "",
    ]

    for table in schema_context.tables:
        est = f" (~{table.row_count_estimate:,} rows)" if table.row_count_estimate else ""
        lines.append(
            f"TABLE: [{table.schema}].[{table.name}]{est}"
        )
        for col in table.columns:
            flags = []
            if col.is_primary_key: flags.append("PK")
            if col.is_foreign_key: flags.append(f"FK→{col.references}")
            if not col.nullable:   flags.append("NOT NULL")
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            sample = ""
            if col.sample_values:
                vals = [repr(v) for v in col.sample_values[:6]]
                sample = f"  e.g. {', '.join(vals)}"
            lines.append(f"  [{col.name}]  {col.raw_type}{flag_str}{sample}")

        rels = schema_context.relationships.get(table.name, [])
        if rels:
            lines.append("  JOINS:")
            for fk_col, ref_tbl, ref_col in rels:
                lines.append(
                    f"    [{table.name}].[{fk_col}] → [{ref_tbl}].[{ref_col}]"
                )
        lines.append("")

    return "\n".join(lines)
