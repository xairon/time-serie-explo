"""
Pydantic schemas for data sources.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field


class DatabaseConnectionRequest(BaseModel):
    """Request to connect to a PostgreSQL database."""
    
    host: str = Field(..., description="Database host")
    port: int = Field(5432, description="Database port")
    database: str = Field(..., description="Database name")
    user: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class DatabaseConnectionResponse(BaseModel):
    """Response for database connection test."""
    
    success: bool
    message: str
    schemas: Optional[list[str]] = None


class TableInfo(BaseModel):
    """Information about a database table."""
    
    name: str
    schema: str = "public"
    type: str = "table"  # table or view
    row_count: Optional[int] = None


class ColumnInfo(BaseModel):
    """Information about a table column."""
    
    name: str
    type: str
    nullable: bool
    is_date: bool = False
    is_numeric: bool = False
    cardinality: Optional[int] = None


class TableSchemaResponse(BaseModel):
    """Response with table schema information."""
    
    table_name: str
    schema: str
    columns: list[ColumnInfo]
    date_columns: list[str]
    dimension_columns: list[str]
    row_count: int


class DataQueryRequest(BaseModel):
    """Request to fetch data from a table."""
    
    # Connection credentials
    host: str = Field(..., description="Database host")
    port: int = Field(5432, description="Database port")
    database: str = Field(..., description="Database name")
    user: str = Field(..., description="Username")
    password: str = Field(..., description="Password")
    
    # Query parameters
    table_name: str
    schema: str = "public"
    columns: Optional[list[str]] = None
    date_column: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    filters: Optional[dict[str, str]] = None
    limit: Optional[int] = None


class DataPreviewResponse(BaseModel):
    """Response with data preview."""
    
    columns: list[str]
    data: list[dict]
    total_rows: int
    preview_rows: int


class StationSummaryRequest(BaseModel):
    """Request params for getting station summary."""
    host: str
    port: int
    database: str
    user: str
    password: str
    schema: str = "public"
    
    table_name: str
    station_column: str
    date_column: Optional[str] = None
    lat_column: Optional[str] = None
    lon_column: Optional[str] = None
    limit: int = 5000


class StationSummaryResponse(BaseModel):
    """Response with station summary."""
    stations: list[dict[str, Any]]
    total_stations: int
