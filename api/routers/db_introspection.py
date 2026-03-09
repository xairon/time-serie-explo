"""Database introspection API router.

Exposes the BRGM PostgreSQL structure (schemas, tables, columns, stations)
so the frontend can build a dynamic import form.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from api.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/db", tags=["db-introspection"])


def _get_brgm_engine():
    """Create a sync SQLAlchemy engine for BRGM introspection."""
    from dashboard.utils.postgres_connector import create_connection

    return create_connection(
        host=settings.brgm_db_host,
        port=settings.brgm_db_port,
        database=settings.brgm_db_name,
        user=settings.brgm_db_user,
        password=settings.brgm_db_password,
    )


@router.get("/schemas")
async def list_schemas() -> list[str]:
    """List available database schemas (filtered to useful ones)."""
    from dashboard.utils.postgres_connector import list_schemas as _list_schemas

    engine = _get_brgm_engine()
    try:
        all_schemas = _list_schemas(engine)
        # Filter out internal/temporary schemas
        skip_prefixes = ("pg_temp", "pg_toast", "_timescaledb", "timescaledb", "toolkit_experimental", "topology")
        return [s for s in all_schemas if not s.startswith(skip_prefixes)]
    finally:
        engine.dispose()


@router.get("/tables")
async def list_tables(schema: str = Query("gold")) -> dict[str, list[str]]:
    """List tables and views in a schema."""
    from dashboard.utils.postgres_connector import list_tables_and_views

    engine = _get_brgm_engine()
    try:
        return list_tables_and_views(engine, schema)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        engine.dispose()


@router.get("/columns")
async def get_columns(
    table: str = Query(...),
    schema: str = Query("gold"),
) -> dict[str, Any]:
    """Get column info, row count, and detected date columns for a table."""
    from dashboard.utils.postgres_connector import (
        get_table_schema,
        get_row_count,
        detect_date_columns,
    )

    engine = _get_brgm_engine()
    try:
        columns = get_table_schema(engine, table, schema)
        row_count = get_row_count(engine, table, schema)
        date_cols = detect_date_columns(engine, table, schema)
        return {
            "columns": columns,
            "row_count": row_count,
            "date_columns": date_cols,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        engine.dispose()


@router.get("/distinct")
async def get_distinct_values(
    table: str = Query(...),
    column: str = Query(...),
    schema: str = Query("gold"),
    limit: Optional[int] = Query(None),
) -> list[Any]:
    """Get distinct values for a column (station codes, etc.)."""
    from dashboard.utils.postgres_connector import get_distinct_values

    engine = _get_brgm_engine()
    try:
        return get_distinct_values(engine, table, column, schema, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        engine.dispose()


@router.get("/date-range")
async def get_date_range(
    table: str = Query(...),
    column: str = Query(...),
    schema: str = Query("gold"),
) -> dict[str, str | None]:
    """Get min/max date range for a date column."""
    from dashboard.utils.postgres_connector import get_date_range

    engine = _get_brgm_engine()
    try:
        min_date, max_date = get_date_range(engine, table, column, schema)
        return {"min": min_date, "max": max_date}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        engine.dispose()
