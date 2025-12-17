"""PostgreSQL connector utility for database import.

Provides functions to connect to PostgreSQL, list tables/views,
get schemas, and fetch data with filters.
"""

import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from urllib.parse import quote_plus


def create_connection(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    connect_timeout: int = 10
) -> Engine:
    """
    Create a SQLAlchemy engine for PostgreSQL connection.
    
    Args:
        host: Database host
        port: Database port
        database: Database name
        user: Username
        password: Password
        connect_timeout: Connection timeout in seconds
        
    Returns:
        SQLAlchemy Engine
    """
    # URL-encode password to handle special characters
    encoded_password = quote_plus(password)
    
    connection_string = (
        f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{database}"
        f"?connect_timeout={connect_timeout}"
    )
    
    engine = create_engine(connection_string, pool_pre_ping=True)
    return engine


def test_connection(engine: Engine) -> Tuple[bool, str]:
    """
    Test if the connection is valid.
    
    Args:
        engine: SQLAlchemy Engine
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            return True, f"Connected! PostgreSQL {version.split(',')[0].replace('PostgreSQL ', '')}"
    except Exception as e:
        return False, str(e)


def list_tables_and_views(engine: Engine, schema: str = "public") -> Dict[str, List[str]]:
    """
    List all tables and views in the database.
    
    Args:
        engine: SQLAlchemy Engine
        schema: Schema name (default: public)
        
    Returns:
        Dict with 'tables' and 'views' lists
    """
    inspector = inspect(engine)
    
    tables = inspector.get_table_names(schema=schema)
    views = inspector.get_view_names(schema=schema)
    
    return {
        'tables': sorted(tables),
        'views': sorted(views)
    }


def list_schemas(engine: Engine) -> List[str]:
    """
    List all schemas in the database.
    
    Args:
        engine: SQLAlchemy Engine
        
    Returns:
        List of schema names
    """
    query = """
        SELECT schema_name 
        FROM information_schema.schemata 
        WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
        ORDER BY schema_name
    """
    
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return [row[0] for row in result.fetchall()]


def get_table_schema(engine: Engine, table_name: str, schema: str = "public") -> List[Dict[str, Any]]:
    """
    Get column information for a table or view.
    
    Args:
        engine: SQLAlchemy Engine
        table_name: Name of the table or view
        schema: Schema name
        
    Returns:
        List of column dictionaries with name, type, nullable
    """
    inspector = inspect(engine)
    
    try:
        columns = inspector.get_columns(table_name, schema=schema)
        return [
            {
                'name': col['name'],
                'type': str(col['type']),
                'nullable': col.get('nullable', True)
            }
            for col in columns
        ]
    except Exception:
        return []


def get_row_count(
    engine: Engine,
    table_name: str,
    schema: str = "public",
    where_clause: Optional[str] = None
) -> int:
    """
    Get the row count of a table with optional filters.
    
    Args:
        engine: SQLAlchemy Engine
        table_name: Name of the table
        schema: Schema name
        where_clause: Optional WHERE clause (without 'WHERE' keyword)
        
    Returns:
        Row count
    """
    query = f'SELECT COUNT(*) FROM "{schema}"."{table_name}"'
    if where_clause:
        query += f" WHERE {where_clause}"
    
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return result.scalar()


def get_distinct_values(
    engine: Engine,
    table_name: str,
    column_name: str,
    schema: str = "public",
    limit: int = 100
) -> List[Any]:
    """
    Get distinct values for a column (for filter dropdowns).
    
    Args:
        engine: SQLAlchemy Engine
        table_name: Name of the table
        column_name: Column to get distinct values from
        schema: Schema name
        limit: Maximum number of values to return
        
    Returns:
        List of distinct values
    """
    query = f'''
        SELECT DISTINCT "{column_name}" 
        FROM "{schema}"."{table_name}" 
        WHERE "{column_name}" IS NOT NULL
        ORDER BY "{column_name}"
        LIMIT {limit}
    '''
    
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return [row[0] for row in result.fetchall()]


def detect_date_columns(engine: Engine, table_name: str, schema: str = "public") -> List[str]:
    """
    Detect columns that are likely date/timestamp columns.
    
    Args:
        engine: SQLAlchemy Engine
        table_name: Name of the table
        schema: Schema name
        
    Returns:
        List of date/timestamp column names
    """
    columns = get_table_schema(engine, table_name, schema)
    
    # SQL date/time types (case-insensitive)
    date_types = ['date', 'timestamp', 'datetime', 'time', 'interval']
    
    # Common date column name patterns
    date_name_patterns = ['date', 'time', 'created', 'updated', 'timestamp', '_at', 'jour', 'day']
    
    date_columns = []
    
    for col in columns:
        col_type = col['type'].lower()
        col_name = col['name'].lower()
        
        # Check by type
        if any(dt in col_type for dt in date_types):
            date_columns.append(col['name'])
        # Check by name pattern (fallback)
        elif any(pattern in col_name for pattern in date_name_patterns):
            date_columns.append(col['name'])
    
    return date_columns


def get_date_range(
    engine: Engine,
    table_name: str,
    date_column: str,
    schema: str = "public"
) -> Tuple[Optional[str], Optional[str]]:
    """
    Get the min and max dates from a date column in the full table.
    
    Args:
        engine: SQLAlchemy Engine
        table_name: Name of the table
        date_column: Name of the date column
        schema: Schema name
        
    Returns:
        Tuple of (min_date, max_date) as strings, or (None, None) if error
    """
    query = f'''
        SELECT 
            MIN("{date_column}")::text,
            MAX("{date_column}")::text
        FROM "{schema}"."{table_name}"
    '''
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            row = result.fetchone()
            if row:
                return row[0], row[1]
    except Exception:
        pass
    
    return None, None


def detect_dimension_columns(
    engine: Engine,
    table_name: str,
    schema: str = "public",
    max_cardinality: int = 100
) -> List[Dict[str, Any]]:
    """
    Detect columns suitable for filtering (low cardinality categorical).
    
    Args:
        engine: SQLAlchemy Engine
        table_name: Name of the table
        schema: Schema name
        max_cardinality: Maximum distinct values to consider as dimension
        
    Returns:
        List of dimension column info with name and cardinality
    """
    columns = get_table_schema(engine, table_name, schema)
    dimensions = []
    
    for col in columns:
        col_type = col['type'].lower()
        # Skip numeric, date, and large text columns
        if any(t in col_type for t in ['int', 'float', 'numeric', 'decimal', 'date', 'time', 'text', 'json']):
            continue
            
        try:
            query = f'''
                SELECT COUNT(DISTINCT "{col['name']}") 
                FROM "{schema}"."{table_name}"
            '''
            with engine.connect() as conn:
                result = conn.execute(text(query))
                cardinality = result.scalar()
                
            if cardinality and cardinality <= max_cardinality:
                dimensions.append({
                    'name': col['name'],
                    'cardinality': cardinality,
                    'type': col['type']
                })
        except Exception:
            continue
    
    return dimensions


def fetch_data(
    engine: Engine,
    table_name: str,
    columns: List[str],
    schema: str = "public",
    date_column: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch data from a table with filters.
    
    Args:
        engine: SQLAlchemy Engine
        table_name: Name of the table
        columns: List of columns to select
        schema: Schema name
        date_column: Column to filter by date range
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        filters: Dict of column -> value(s) for WHERE clauses
        limit: Optional row limit
        
    Returns:
        DataFrame with the queried data
    """
    # Build column list
    cols_str = ", ".join([f'"{c}"' for c in columns])
    
    # Build query
    query = f'SELECT {cols_str} FROM "{schema}"."{table_name}"'
    
    # Build WHERE clauses
    where_parts = []
    params = {}
    
    # Date range filter
    if date_column and start_date:
        where_parts.append(f'"{date_column}" >= :start_date')
        params['start_date'] = start_date
    
    if date_column and end_date:
        where_parts.append(f'"{date_column}" <= :end_date')
        params['end_date'] = end_date
    
    # Dimension filters
    if filters:
        for col, values in filters.items():
            if values is None or (isinstance(values, list) and len(values) == 0):
                continue
            
            if isinstance(values, list):
                # Multiple values - use IN clause
                placeholders = ", ".join([f":filter_{col}_{i}" for i in range(len(values))])
                where_parts.append(f'"{col}" IN ({placeholders})')
                for i, v in enumerate(values):
                    params[f"filter_{col}_{i}"] = v
            else:
                # Single value
                where_parts.append(f'"{col}" = :filter_{col}')
                params[f"filter_{col}"] = values
    
    if where_parts:
        query += " WHERE " + " AND ".join(where_parts)
    
    # Add ORDER BY if date column exists
    if date_column:
        query += f' ORDER BY "{date_column}"'
    
    # Add LIMIT if specified
    if limit:
        query += f" LIMIT {limit}"
    
    # Execute and return DataFrame
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
    
    return df


def build_query_preview(
    table_name: str,
    columns: List[str],
    schema: str = "public",
    date_column: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None
) -> str:
    """
    Build a preview of the SQL query that will be executed.
    
    Returns:
        SQL query string for display
    """
    cols_str = ", ".join([f'"{c}"' for c in columns])
    query = f'SELECT {cols_str}\nFROM "{schema}"."{table_name}"'
    
    where_parts = []
    
    if date_column and start_date:
        where_parts.append(f'"{date_column}" >= \'{start_date}\'')
    
    if date_column and end_date:
        where_parts.append(f'"{date_column}" <= \'{end_date}\'')
    
    if filters:
        for col, values in filters.items():
            if values is None or (isinstance(values, list) and len(values) == 0):
                continue
            
            if isinstance(values, list):
                vals_str = ", ".join([f"'{v}'" for v in values])
                where_parts.append(f'"{col}" IN ({vals_str})')
            else:
                where_parts.append(f'"{col}" = \'{values}\'')
    
    if where_parts:
        query += "\nWHERE " + "\n  AND ".join(where_parts)
    
    if date_column:
        query += f'\nORDER BY "{date_column}"'
    
    if limit:
        query += f"\nLIMIT {limit}"
    
    return query
