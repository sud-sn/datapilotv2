# DataPilot – Phase 1

## Quick Start

```bash
# 1. Install Ollama models (air-gapped)
ollama pull sqlcoder:15b
ollama pull qwen2.5-coder:7b

# 2. Install ODBC Driver 18 for SQL Server (for Azure SQL)
# macOS:  brew install msodbcsql18
# Ubuntu: follow https://learn.microsoft.com/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server

# 3. Install Python deps
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env with your DB credentials

# 5. Run
python main.py
```

## Architecture

```
User NL Query
     │
     ▼
POST /query  { query: "...", db_source: "postgresql" | "azure_sql" }
     │
     ▼
NLPToSQLPipeline
  ├── get_schema_cached(adapter)          ← introspects DB once, caches 1hr
  ├── KnowledgeBase.format_prompt(schema) ← loads dialect-specific KB + schema
  ├── OllamaProvider.generate_sql(query)  ← routes to 7b or 15b by complexity
  ├── SQLValidator.validate(sql)          ← 3-layer defence
  └── adapter.execute(canonical_sql)      ← runs against PostgreSQL or Azure SQL
```

## Model Routing

| Query Complexity | Model Used         | Example |
|------------------|--------------------|---------|
| Simple           | qwen2.5-coder:7b   | "count orders today" |
| Medium/Complex   | sqlcoder:15b        | "revenue by region last 30 days with joins" |

Set `DATAPILOT_LLM_AUTO_ROUTE=false` to always use the primary model.

## API Endpoints

| Method | Path              | Description                       |
|--------|-------------------|-----------------------------------|
| GET    | /health           | DB + model health check           |
| GET    | /sources          | List configured databases         |
| POST   | /query            | NLP → SQL → Results               |
| POST   | /schema/refresh   | Force schema cache refresh        |
