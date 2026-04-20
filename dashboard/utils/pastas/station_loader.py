"""Load station time series from the BRGM data warehouse."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


@dataclass
class StationSeries:
    code_bss: str
    piezo: pd.Series      # niveau_nappe_eau, index=date
    precip: pd.Series     # total_precipitation (ERA5), index=date
    evap: pd.Series       # potential_evaporation (ERA5), index=date
    metadata: dict        # nom_commune, departement, lat/lon, etc.


def load_station_series(code_bss: str, db_url: str) -> StationSeries:
    """Fetch piezo + climate series for a station from the gold schema.

    Args:
        code_bss: BSS station code (e.g. "BSS001ABCD")
        db_url: SQLAlchemy connection string to brgm-postgres

    Returns:
        StationSeries with piezo, precip, evap as pd.Series indexed by date.

    Raises:
        ValueError: if station not found or insufficient data.
    """
    engine = create_engine(db_url)
    try:
        query = text("""
            SELECT date, niveau_nappe_eau, total_precipitation, potential_evaporation,
                   nom_commune, code_departement, nom_departement,
                   station_latitude, station_longitude, altitude_station
            FROM gold.hubeau_daily_chroniques
            WHERE code_bss = :code_bss
            ORDER BY date
        """)

        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"code_bss": code_bss}, parse_dates=["date"])

        if df.empty:
            raise ValueError(f"No data found for station {code_bss}")

        df = df.set_index("date").sort_index()

        # Extract metadata from first row
        meta_row = df.iloc[0]
        metadata = {
            "nom_commune": str(meta_row.get("nom_commune", "")),
            "code_departement": str(meta_row.get("code_departement", "")),
            "nom_departement": str(meta_row.get("nom_departement", "")),
            "latitude": float(meta_row["station_latitude"]) if pd.notna(meta_row.get("station_latitude")) else None,
            "longitude": float(meta_row["station_longitude"]) if pd.notna(meta_row.get("station_longitude")) else None,
            "altitude": float(meta_row["altitude_station"]) if pd.notna(meta_row.get("altitude_station")) else None,
        }

        piezo = df["niveau_nappe_eau"].dropna()
        precip = df["total_precipitation"].dropna()
        evap = df["potential_evaporation"].dropna()

        piezo.name = "piezo"
        precip.name = "precip"
        evap.name = "evap"

        return StationSeries(
            code_bss=code_bss,
            piezo=piezo,
            precip=precip,
            evap=evap,
            metadata=metadata,
        )
    finally:
        engine.dispose()
