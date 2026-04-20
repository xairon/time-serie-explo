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
        # 1. Piezo from hubeau_daily_chroniques (irregular, Pastas handles it)
        piezo_query = text("""
            SELECT date, niveau_nappe_eau
            FROM gold.hubeau_daily_chroniques
            WHERE code_bss = :code_bss AND niveau_nappe_eau IS NOT NULL
            ORDER BY date
        """)

        # 2. Station metadata + ERA5 grid point from mapping table
        meta_query = text("""
            SELECT era5_latitude, era5_longitude,
                   station_latitude, station_longitude, altitude_station,
                   nom_commune, code_departement, nom_departement
            FROM gold.int_station_era5_mapping
            WHERE code_bss = :code_bss
            LIMIT 1
        """)

        with engine.connect() as conn:
            piezo_df = pd.read_sql(piezo_query, conn, params={"code_bss": code_bss}, parse_dates=["date"])
            meta_df = pd.read_sql(meta_query, conn, params={"code_bss": code_bss})

        if piezo_df.empty:
            raise ValueError(f"No piezometric data for station {code_bss}")
        if meta_df.empty:
            raise ValueError(f"No ERA5 mapping for station {code_bss}")

        meta_row = meta_df.iloc[0]
        era5_lat = float(meta_row["era5_latitude"])
        era5_lon = float(meta_row["era5_longitude"])

        metadata = {
            "nom_commune": str(meta_row.get("nom_commune", "")),
            "code_departement": str(meta_row.get("code_departement", "")),
            "nom_departement": str(meta_row.get("nom_departement", "")),
            "latitude": float(meta_row["station_latitude"]) if pd.notna(meta_row.get("station_latitude")) else None,
            "longitude": float(meta_row["station_longitude"]) if pd.notna(meta_row.get("station_longitude")) else None,
            "altitude": float(meta_row["altitude_station"]) if pd.notna(meta_row.get("altitude_station")) else None,
        }

        # 3. ERA5 climate data — continuous daily from dedicated table
        era5_query = text("""
            SELECT era5_date AS date, total_precipitation, potential_evaporation
            FROM gold.int_era5_for_all_stations
            WHERE latitude = :lat AND longitude = :lon
            ORDER BY era5_date
        """)

        with engine.connect() as conn:
            era5_df = pd.read_sql(era5_query, conn, params={"lat": era5_lat, "lon": era5_lon}, parse_dates=["date"])

        if era5_df.empty:
            raise ValueError(f"No ERA5 data for station {code_bss} (grid {era5_lat}, {era5_lon})")

        # Piezo: irregular index, Pastas handles it natively
        piezo = piezo_df.set_index("date")["niveau_nappe_eau"].sort_index()
        piezo = piezo[~piezo.index.duplicated(keep="first")]
        piezo.name = "piezo"

        # ERA5 stresses: should be daily and continuous
        era5_df = era5_df.set_index("date").sort_index()
        era5_df = era5_df[~era5_df.index.duplicated(keep="first")]

        precip = era5_df["total_precipitation"]
        precip.name = "precip"
        # Ensure positive values (ERA5 precip is always >= 0)
        precip = precip.clip(lower=0)

        evap = era5_df["potential_evaporation"]
        evap.name = "evap"
        # ERA5 potential_evaporation is negative (energy convention) — flip sign
        evap = (-evap).clip(lower=0)

        # Set freq explicitly if pandas can't infer it
        if precip.index.freq is None:
            precip = precip.asfreq("D")
        if evap.index.freq is None:
            evap = evap.asfreq("D")

        return StationSeries(
            code_bss=code_bss,
            piezo=piezo,
            precip=precip,
            evap=evap,
            metadata=metadata,
        )
    finally:
        engine.dispose()
