"""Load station time series from the BRGM data warehouse."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

_MIN_COVERAGE = 0.80


@dataclass
class StationSeries:
    code_bss: str
    piezo: pd.Series      # niveau_nappe_eau, index=date
    precip: pd.Series     # total_precipitation (ERA5), index=date, freq='D'
    evap: pd.Series       # potential_evaporation (ERA5), index=date, freq='D'
    temp: pd.Series       # temperature_2m (ERA5), index=date, freq='D'
    metadata: dict        # nom_commune, departement, lat/lon, etc.


def _regularize(s: pd.Series) -> pd.Series:
    """Resample to daily, fill all gaps (interpolate short, ffill long), trim edges."""
    s = s[~s.index.duplicated(keep="first")].sort_index()
    s = s.asfreq("D")
    s = s.interpolate(method="linear", limit=7)
    s = s.ffill().bfill()
    first_valid = s.first_valid_index()
    last_valid = s.last_valid_index()
    if first_valid is None:
        return s
    return s.loc[first_valid:last_valid]


def load_station_series(code_bss: str, db_url: str) -> StationSeries:
    """Fetch piezo + climate series for a station from the gold schema.

    Strategy:
    1. Load everything from hubeau_daily_chroniques (piezo + ERA5 joined).
       For well-instrumented stations this gives quasi-daily coverage.
    2. Regularize precip/evap to daily with short-gap interpolation.
    3. If coverage is too low (<80%), fall back to the continuous
       int_era5_for_all_stations table via the station-ERA5 mapping.
    """
    engine = create_engine(db_url)
    try:
        # --- Load from hubeau_daily_chroniques ---
        query = text("""
            SELECT date, niveau_nappe_eau, total_precipitation, potential_evaporation,
                   temperature_2m, nom_commune, code_departement, nom_departement,
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
        df = df[~df.index.duplicated(keep="first")]

        # Metadata from first row
        meta_row = df.iloc[0]
        metadata = {
            "nom_commune": str(meta_row.get("nom_commune", "")),
            "code_departement": str(meta_row.get("code_departement", "")),
            "nom_departement": str(meta_row.get("nom_departement", "")),
            "latitude": float(meta_row["station_latitude"]) if pd.notna(meta_row.get("station_latitude")) else None,
            "longitude": float(meta_row["station_longitude"]) if pd.notna(meta_row.get("station_longitude")) else None,
            "altitude": float(meta_row["altitude_station"]) if pd.notna(meta_row.get("altitude_station")) else None,
        }

        # Piezo: irregular, Pastas handles it
        piezo = df["niveau_nappe_eau"].dropna()
        piezo.name = "piezo"

        # Try regularizing precip/evap from the joined data
        raw_precip = df["total_precipitation"].dropna()
        raw_evap = df["potential_evaporation"].dropna()

        if len(raw_precip) > 1:
            expected_days = (raw_precip.index.max() - raw_precip.index.min()).days + 1
            coverage = len(raw_precip) / expected_days if expected_days > 0 else 0
        else:
            coverage = 0

        if coverage >= _MIN_COVERAGE and len(raw_precip) >= 365:
            logger.info("Station %s: using inline ERA5 (coverage %.0f%%)", code_bss, coverage * 100)
            precip = _regularize(raw_precip.clip(lower=0))
            evap = _regularize((-raw_evap).clip(lower=0))
            raw_temp = df["temperature_2m"].dropna()
            temp = _regularize(raw_temp) if len(raw_temp) > 1 else raw_temp
        else:
            logger.info("Station %s: inline coverage %.0f%%, falling back to ERA5 table", code_bss, coverage * 100)
            precip, evap, temp = _load_era5_fallback(code_bss, engine)

        precip.name = "precip"
        evap.name = "evap"
        temp.name = "temp"

        return StationSeries(
            code_bss=code_bss,
            piezo=piezo,
            precip=precip,
            evap=evap,
            temp=temp,
            metadata=metadata,
        )
    finally:
        engine.dispose()


def _load_era5_fallback(code_bss: str, engine) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Load continuous ERA5 data via the station-grid mapping table."""
    meta_query = text("""
        SELECT era5_latitude, era5_longitude
        FROM gold.int_station_era5_mapping
        WHERE code_bss = :code_bss
        LIMIT 1
    """)

    with engine.connect() as conn:
        meta_df = pd.read_sql(meta_query, conn, params={"code_bss": code_bss})

    if meta_df.empty:
        raise ValueError(f"No ERA5 mapping for station {code_bss}")

    era5_lat = float(meta_df.iloc[0]["era5_latitude"])
    era5_lon = float(meta_df.iloc[0]["era5_longitude"])

    era5_query = text("""
        SELECT era5_date AS date, total_precipitation, potential_evaporation, temperature_2m
        FROM gold.int_era5_for_all_stations
        WHERE latitude = :lat AND longitude = :lon
        ORDER BY era5_date
    """)

    with engine.connect() as conn:
        era5_df = pd.read_sql(era5_query, conn, params={"lat": era5_lat, "lon": era5_lon}, parse_dates=["date"])

    if era5_df.empty:
        raise ValueError(f"No ERA5 data for station {code_bss} (grid {era5_lat}, {era5_lon})")

    era5_df = era5_df.set_index("date").sort_index()
    era5_df = era5_df[~era5_df.index.duplicated(keep="first")]

    precip = era5_df["total_precipitation"].clip(lower=0)
    evap = (-era5_df["potential_evaporation"]).clip(lower=0)
    temp = era5_df["temperature_2m"]

    for s in (precip, evap, temp):
        if s.index.freq is None:
            s = s.asfreq("D")

    return precip, evap, temp
