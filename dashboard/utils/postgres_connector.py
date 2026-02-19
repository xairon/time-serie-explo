"""PostgreSQL connector utility for database import.

Provides functions to connect to PostgreSQL, list tables/views,
get schemas, and fetch data with filters.
"""

import re
import logging
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.engine import URL
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# Regex for safe SQL identifiers (schema, table, column names)
_SAFE_IDENTIFIER_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def _validate_identifier(name: str, kind: str = "identifier") -> str:
    """Validate a SQL identifier to prevent injection.

    Args:
        name: The identifier to validate
        kind: Description for error messages (e.g., 'table', 'schema', 'column')

    Returns:
        The validated identifier

    Raises:
        ValueError: If the identifier contains unsafe characters
    """
    if not name or not _SAFE_IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Invalid SQL {kind}: {name!r}. "
            f"Only alphanumeric characters and underscores are allowed, "
            f"and it must start with a letter or underscore."
        )
    return name


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
    # Use SQLAlchemy URL.create for safe connection string building
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=user,
        password=password,
        host=host,
        port=port,
        database=database,
        query={"connect_timeout": str(connect_timeout)},
    )

    engine = create_engine(url, pool_pre_ping=True)
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
    schema = _validate_identifier(schema, "schema")
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
    schema = _validate_identifier(schema, "schema")
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
    Get the row count of a table.

    Args:
        engine: SQLAlchemy Engine
        table_name: Name of the table
        schema: Schema name
        where_clause: Deprecated, ignored for security. Use fetch_data with filters instead.

    Returns:
        Row count
    """
    schema = _validate_identifier(schema, "schema")
    table_name = _validate_identifier(table_name, "table")

    if where_clause:
        logger.warning("get_row_count: where_clause parameter is ignored for security. Use fetch_data with parameterized filters.")

    query = f'SELECT COUNT(*) FROM "{schema}"."{table_name}"'

    with engine.connect() as conn:
        result = conn.execute(text(query))
        return result.scalar()


def get_distinct_values(
    engine: Engine,
    table_name: str,
    column_name: str,
    schema: str = "public",
    limit: int = None
) -> List[Any]:
    """
    Get distinct values for a column (for filter dropdowns).

    Args:
        engine: SQLAlchemy Engine
        table_name: Name of the table
        column_name: Column to get distinct values from
        schema: Schema name
        limit: Maximum number of values to return (None = no limit)

    Returns:
        List of distinct values
    """
    schema = _validate_identifier(schema, "schema")
    table_name = _validate_identifier(table_name, "table")
    column_name = _validate_identifier(column_name, "column")

    query = f'''
        SELECT DISTINCT "{column_name}"
        FROM "{schema}"."{table_name}"
        WHERE "{column_name}" IS NOT NULL
        ORDER BY "{column_name}"
    '''
    if limit is not None and limit > 0:
        query += f' LIMIT {int(limit)}'

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
    schema = _validate_identifier(schema, "schema")
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
    schema = _validate_identifier(schema, "schema")
    table_name = _validate_identifier(table_name, "table")
    date_column = _validate_identifier(date_column, "column")

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
    schema = _validate_identifier(schema, "schema")
    table_name = _validate_identifier(table_name, "table")

    columns = get_table_schema(engine, table_name, schema)
    dimensions = []

    for col in columns:
        col_type = col['type'].lower()
        # Skip numeric, date, and large text columns
        if any(t in col_type for t in ['int', 'float', 'numeric', 'decimal', 'date', 'time', 'text', 'json']):
            continue

        try:
            col_name = _validate_identifier(col['name'], "column")
            query = f'''
                SELECT COUNT(DISTINCT "{col_name}")
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


def get_station_summary(
    engine: Engine,
    table_name: str,
    station_column: str,
    date_column: Optional[str] = None,
    lat_column: Optional[str] = None,
    lon_column: Optional[str] = None,
    schema: str = "public",
    limit: int = 5000
) -> pd.DataFrame:
    """
    Get summary of stations with row counts and optional date/coordinate info.
    
    This is a fast aggregation query that avoids loading all data.
    
    Args:
        engine: SQLAlchemy Engine
        table_name: Name of the table
        station_column: Column containing station identifiers
        date_column: Optional date column for date range info
        lat_column: Optional latitude column
        lon_column: Optional longitude column
        schema: Schema name
        limit: Maximum number of stations to return
        
    Returns:
        DataFrame with station, row_count, and optional metadata
    """
    schema = _validate_identifier(schema, "schema")
    table_name = _validate_identifier(table_name, "table")
    station_column = _validate_identifier(station_column, "column")

    select_parts = [
        f'"{station_column}" as station',
        'COUNT(*) as row_count'
    ]
    
    if date_column:
        date_column = _validate_identifier(date_column, "column")
        select_parts.extend([
            f'MIN("{date_column}")::text as min_date',
            f'MAX("{date_column}")::text as max_date'
        ])

    if lat_column and lon_column:
        lat_column = _validate_identifier(lat_column, "column")
        lon_column = _validate_identifier(lon_column, "column")
        select_parts.extend([
            f'MIN("{lat_column}") as latitude',
            f'MIN("{lon_column}") as longitude'
        ])
    
    query = f'''
        SELECT {", ".join(select_parts)}
        FROM "{schema}"."{table_name}"
        WHERE "{station_column}" IS NOT NULL
        GROUP BY "{station_column}"
        ORDER BY "{station_column}"
        LIMIT {int(limit)}
    '''

    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


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
    # Validate identifiers
    schema = _validate_identifier(schema, "schema")
    table_name = _validate_identifier(table_name, "table")
    for c in columns:
        _validate_identifier(c, "column")
    if date_column:
        date_column = _validate_identifier(date_column, "column")

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
            col = _validate_identifier(col, "column")
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
        query += f" LIMIT {int(limit)}"
    
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

    NOTE: This function is for DISPLAY ONLY. Never execute the returned string.

    Returns:
        SQL query string for display
    """
    # Validate identifiers
    schema = _validate_identifier(schema, "schema")
    table_name = _validate_identifier(table_name, "table")
    for c in columns:
        _validate_identifier(c, "column")
    if date_column:
        date_column = _validate_identifier(date_column, "column")

    def _escape_value(v) -> str:
        """Escape single quotes for display only."""
        return str(v).replace("'", "''")

    cols_str = ", ".join([f'"{c}"' for c in columns])
    query = f'SELECT {cols_str}\nFROM "{schema}"."{table_name}"'

    where_parts = []

    if date_column and start_date:
        where_parts.append(f'"{date_column}" >= \'{_escape_value(start_date)}\'')

    if date_column and end_date:
        where_parts.append(f'"{date_column}" <= \'{_escape_value(end_date)}\'')

    if filters:
        for col, values in filters.items():
            col = _validate_identifier(col, "column")
            if values is None or (isinstance(values, list) and len(values) == 0):
                continue

            if isinstance(values, list):
                vals_str = ", ".join([f"'{_escape_value(v)}'" for v in values])
                where_parts.append(f'"{col}" IN ({vals_str})')
            else:
                where_parts.append(f'"{col}" = \'{_escape_value(values)}\'')

    if where_parts:
        query += "\nWHERE " + "\n  AND ".join(where_parts)

    if date_column:
        query += f'\nORDER BY "{date_column}"'

    if limit:
        query += f"\nLIMIT {int(limit)}"
    
    return query
