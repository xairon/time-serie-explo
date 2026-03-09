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


@router.get("/stations/search")
async def search_stations(
    q: str = Query("", description="Search term (code BSS, commune, département)"),
    departement: Optional[str] = Query(None),
    tendance: Optional[str] = Query(None),
    alerte: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    """Search piezo stations with metadata from dim_piezo_stations."""
    from sqlalchemy import text as sql_text

    engine = _get_brgm_engine()
    try:
        where_parts = []
        params: dict[str, Any] = {"lim": limit}

        if q and len(q) >= 2:
            where_parts.append(
                "(code_bss ILIKE :q OR nom_commune ILIKE :q OR nom_departement ILIKE :q OR codes_bdlisa ILIKE :q)"
            )
            params["q"] = f"%{q}%"

        if departement:
            where_parts.append("code_departement = :dept")
            params["dept"] = departement

        if tendance:
            where_parts.append("tendance_classification = :tendance")
            params["tendance"] = tendance

        if alerte:
            where_parts.append("niveau_alerte = :alerte")
            params["alerte"] = alerte

        where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        query = f"""
            SELECT code_bss, nom_commune, code_departement, nom_departement,
                   codes_bdlisa, altitude_station, latitude, longitude,
                   premiere_mesure::text, derniere_mesure::text,
                   nb_mesures_total, niveau_moyen_global,
                   amplitude_totale, tendance_classification,
                   niveau_alerte, classification_derniere_annee,
                   qualite_tendance
            FROM gold.dim_piezo_stations
            {where_sql}
            ORDER BY nb_mesures_total DESC NULLS LAST
            LIMIT :lim
        """

        with engine.connect() as conn:
            result = conn.execute(sql_text(query), params)
            rows = [dict(r._mapping) for r in result.fetchall()]

        # Also get filter options (cached-friendly)
        return {"stations": rows, "total": len(rows)}
    finally:
        engine.dispose()


@router.get("/stations/filters")
async def station_filters() -> dict[str, list[str]]:
    """Get available filter values for station search."""
    from sqlalchemy import text as sql_text

    engine = _get_brgm_engine()
    try:
        filters: dict[str, list[str]] = {}
        with engine.connect() as conn:
            for col, key in [
                ("code_departement", "departements"),
                ("tendance_classification", "tendances"),
                ("niveau_alerte", "alertes"),
                ("classification_derniere_annee", "classifications"),
            ]:
                result = conn.execute(
                    sql_text(
                        f'SELECT DISTINCT "{col}" FROM gold.dim_piezo_stations '
                        f'WHERE "{col}" IS NOT NULL ORDER BY 1'
                    )
                )
                filters[key] = [r[0] for r in result.fetchall()]
        return filters
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
