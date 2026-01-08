"""
Data Sources endpoints - PostgreSQL connection and CSV upload.

Reuses logic from dashboard/utils/postgres_connector.py
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from app.schemas import (
    DatabaseConnectionRequest,
    DatabaseConnectionResponse,
    TableInfo,
    ColumnInfo,
    TableSchemaResponse,
    DataQueryRequest,
    DataPreviewResponse,
    StationSummaryRequest,
    StationSummaryResponse,
)
from app.config import get_settings

# Import from existing utils (core)
import sys
from pathlib import Path

from starlette.concurrency import run_in_threadpool
import logging

logger = logging.getLogger(__name__)

# Add legacy dashboard to path for now (will be migrated to core/)
sys.path.insert(0, str(Path(__file__).parents[5]))

router = APIRouter()


@router.post("/db/connect", response_model=DatabaseConnectionResponse)
async def connect_database(request: DatabaseConnectionRequest):
    """
    Test connection to a PostgreSQL database.
    
    Returns list of available schemas on success.
    """
    try:
        from core.database import create_connection, test_connection, list_schemas
        
        print(f"DEBUG: Processing connection request: {request.host}:{request.port}")
        logger.info(f"Connecting to DB: {request.host}:{request.port}/{request.database} as {request.user}")
        
        # Run synchronous create_connection in threadpool
        print("DEBUG: Calling create_connection in threadpool")
        engine = await run_in_threadpool(
            create_connection,
            host=request.host,
            port=request.port,
            database=request.database,
            user=request.user,
            password=request.password,
        )
        
        # Run synchronous test_connection in threadpool
        success, message = await run_in_threadpool(test_connection, engine)
        
        if success:
            schemas = await run_in_threadpool(list_schemas, engine)
            return DatabaseConnectionResponse(
                success=True,
                message="Connection successful",
                schemas=schemas,
            )
        else:
            logger.error(f"Connection failed: {message}")
            return DatabaseConnectionResponse(
                success=False,
                message=message,
            )
    except Exception as e:
        logger.exception("Exception in connect_database")
        return DatabaseConnectionResponse(
            success=False,
            message=f"Connection failed: {str(e)}",
        )


@router.get("/db/tables")
async def list_tables(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    schema: str = "public",
):
    """
    List all tables and views in a database schema.
    """
    try:
        from core.database import create_connection, list_tables_and_views
        
        engine = await run_in_threadpool(
            create_connection,
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
        )
        
        result = await run_in_threadpool(list_tables_and_views, engine, schema=schema)
        
        tables = [
            TableInfo(name=t, schema=schema, type="table")
            for t in result.get("tables", [])
        ]
        views = [
            TableInfo(name=v, schema=schema, type="view")
            for v in result.get("views", [])
        ]
        
        return {"tables": tables, "views": views}
    except Exception as e:
        logger.exception("Exception in list_tables")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/db/schema/{table_name}", response_model=TableSchemaResponse)
async def get_table_schema(
    table_name: str,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    schema: str = "public",
):
    """
    Get column information for a table or view.
    """
    try:
        from core.database import (
            create_connection,
            get_table_schema as get_schema,
            detect_date_columns,
            detect_dimension_columns,
            get_row_count,
        )
        
        engine = create_connection(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
        )
        
        columns_raw = get_schema(engine, table_name, schema=schema)
        date_cols = detect_date_columns(engine, table_name, schema=schema)
        dim_cols = detect_dimension_columns(engine, table_name, schema=schema)
        row_count = get_row_count(engine, table_name, schema=schema)
        
        numeric_types = {"integer", "bigint", "float", "double precision", "numeric", "real"}
        
        columns = [
            ColumnInfo(
                name=col["name"],
                type=str(col["type"]),
                nullable=col.get("nullable", True),
                is_date=col["name"] in date_cols,
                is_numeric=str(col["type"]).lower() in numeric_types,
            )
            for col in columns_raw
        ]
        
        return TableSchemaResponse(
            table_name=table_name,
            schema=schema,
            columns=columns,
            date_columns=date_cols,
            dimension_columns=[d["name"] for d in dim_cols],
            row_count=row_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/db/query", response_model=DataPreviewResponse)
async def query_data(request: DataQueryRequest):
    """
    Fetch data from a table with filters.
    """
    try:
        from core.database import create_connection, fetch_data, get_table_schema
        
        # Use credentials from request
        engine = create_connection(
            host=request.host,
            port=request.port,
            database=request.database,
            user=request.user,
            password=request.password,
        )
        
        # If no columns specified, get all columns from table schema
        columns = request.columns
        if not columns:
            schema_info = get_table_schema(engine, request.table_name, request.schema)
            columns = [col['name'] for col in schema_info]
        
        if not columns:
            raise HTTPException(status_code=400, detail="Could not determine columns for table")
        
        df = fetch_data(
            engine=engine,
            table_name=request.table_name,
            columns=columns,
            schema=request.schema,
            date_column=request.date_column,
            start_date=request.start_date,
            end_date=request.end_date,
            filters=request.filters,
            limit=request.limit or 1000,
        )
        
        return DataPreviewResponse(
            columns=df.columns.tolist(),
            data=df.to_dict(orient="records"),
            total_rows=len(df),
            preview_rows=len(df),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/db/stations", response_model=StationSummaryResponse)
async def get_stations(request: StationSummaryRequest):
    """
    Get summary of all stations with coordinates.
    """
    try:
        from core.database import create_connection, get_station_summary
        
        # Use credentials from request
        engine = create_connection(
            host=request.host,
            port=request.port,
            database=request.database,
            user=request.user,
            password=request.password,
        )
        
        df = get_station_summary(
            engine=engine,
            table_name=request.table_name,
            station_column=request.station_column,
            date_column=request.date_column,
            lat_column=request.lat_column,
            lon_column=request.lon_column,
            schema=request.schema,
            limit=request.limit,
        )
        
        return StationSummaryResponse(
            stations=df.to_dict(orient="records"),
            total_stations=len(df),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file for dataset creation.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    settings = get_settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = settings.uploads_dir / file.filename
    
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Quick validation
        import pandas as pd
        df = pd.read_csv(file_path, nrows=5)
        
        return {
            "filename": file.filename,
            "path": str(file_path),
            "columns": df.columns.tolist(),
            "preview_rows": 5,
        }
    except Exception as e:
        # Clean up on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
