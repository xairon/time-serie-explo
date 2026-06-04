"""Load station time series from the BRGM data warehouse."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

_MIN_COVERAGE = 0.80


@dataclass
class StationSeries:
    code_bss: str
    piezo: pd.Series
    precip: pd.Series
    evap: pd.Series
    temp: pd.Series
    metadata: dict[str, Any]


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


def _safe_float(val) -> float | None:
    if val is None or pd.isna(val):
        return None
    return float(val)


def _safe_str(val) -> str | None:
    if val is None or pd.isna(val):
        return None
    return str(val)


def load_station_metadata(code_bss: str, db_url: str) -> dict[str, Any]:
    """Fast metadata-only query from dim_piezo_stations + BDLISA mapping.

    Returns in <50ms — no heavy time series loading.
    """
    engine = create_engine(db_url)
    try:
        query = text("""
            SELECT s.code_bss, s.nom_commune, s.code_departement, s.nom_departement,
                   s.latitude, s.longitude, s.altitude_station,
                   s.premiere_mesure, s.derniere_mesure,
                   s.nb_mesures_total, s.nb_mois_total,
                   s.niveau_moyen_global, s.niveau_min_absolu, s.niveau_max_absolu,
                   s.niveau_stddev_global, s.amplitude_totale,
                   s.tendance_classification, s.niveau_alerte,
                   s.qualite_tendance, s.slope_niveau,
                   s.precipitation_moyenne_mensuelle, s.temperature_moyenne_globale,
                   m.codes_bdlisa, m.nature_eh, m.milieu_eh
            FROM gold.dim_piezo_stations s
            LEFT JOIN gold.int_station_era5_mapping m ON m.code_bss = s.code_bss
            WHERE s.code_bss = :code_bss
            LIMIT 1
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"code_bss": code_bss})

        if df.empty:
            raise ValueError(f"Station {code_bss} not found in dim_piezo_stations")

        r = df.iloc[0]
        return {
            "code_bss": code_bss,
            "nom_commune": _safe_str(r.get("nom_commune")),
            "code_departement": _safe_str(r.get("code_departement")),
            "nom_departement": _safe_str(r.get("nom_departement")),
            "latitude": _safe_float(r.get("latitude")),
            "longitude": _safe_float(r.get("longitude")),
            "altitude": _safe_float(r.get("altitude_station")),
            "premiere_mesure": str(r["premiere_mesure"]) if pd.notna(r.get("premiere_mesure")) else None,
            "derniere_mesure": str(r["derniere_mesure"]) if pd.notna(r.get("derniere_mesure")) else None,
            "nb_mesures_total": int(r["nb_mesures_total"]) if pd.notna(r.get("nb_mesures_total")) else None,
            "nb_mois_total": int(r["nb_mois_total"]) if pd.notna(r.get("nb_mois_total")) else None,
            "niveau_moyen_global": _safe_float(r.get("niveau_moyen_global")),
            "niveau_min_absolu": _safe_float(r.get("niveau_min_absolu")),
            "niveau_max_absolu": _safe_float(r.get("niveau_max_absolu")),
            "niveau_stddev_global": _safe_float(r.get("niveau_stddev_global")),
            "amplitude_totale": _safe_float(r.get("amplitude_totale")),
            "tendance_classification": _safe_str(r.get("tendance_classification")),
            "niveau_alerte": _safe_str(r.get("niveau_alerte")),
            "qualite_tendance": _safe_str(r.get("qualite_tendance")),
            "slope_niveau": _safe_float(r.get("slope_niveau")),
            "precipitation_moyenne_mensuelle": _safe_float(r.get("precipitation_moyenne_mensuelle")),
            "temperature_moyenne_globale": _safe_float(r.get("temperature_moyenne_globale")),
            "codes_bdlisa": _safe_str(r.get("codes_bdlisa")),
            "nature_eh": _safe_str(r.get("nature_eh")),
            "milieu_eh": _safe_str(r.get("milieu_eh")),
        }
    finally:
        engine.dispose()


def load_station_series(code_bss: str, db_url: str) -> StationSeries:
    """Fetch piezo + climate series for a station from the gold schema.

    Single merged query with LEFT JOIN for BDLISA metadata.
    Falls back to ERA5 grid table if inline coverage < 80%.
    """
    engine = create_engine(db_url)
    try:
        query = text("""
            SELECT d.date, d.niveau_nappe_eau, d.total_precipitation,
                   d.potential_evaporation, d.temperature_2m,
                   d.nom_commune, d.code_departement, d.nom_departement,
                   d.station_latitude, d.station_longitude, d.altitude_station,
                   m.codes_bdlisa, m.nature_eh, m.milieu_eh
            FROM gold.hubeau_daily_chroniques d
            LEFT JOIN gold.int_station_era5_mapping m ON m.code_bss = d.code_bss
            WHERE d.code_bss = :code_bss
            ORDER BY d.date
        """)

        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"code_bss": code_bss}, parse_dates=["date"])

        if df.empty:
            raise ValueError(f"No data found for station {code_bss}")

        df = df.set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="first")]

        meta_row = df.iloc[0]
        metadata: dict[str, Any] = {
            "nom_commune": _safe_str(meta_row.get("nom_commune")),
            "code_departement": _safe_str(meta_row.get("code_departement")),
            "nom_departement": _safe_str(meta_row.get("nom_departement")),
            "latitude": _safe_float(meta_row.get("station_latitude")),
            "longitude": _safe_float(meta_row.get("station_longitude")),
            "altitude": _safe_float(meta_row.get("altitude_station")),
            "codes_bdlisa": _safe_str(meta_row.get("codes_bdlisa")),
            "nature_eh": _safe_str(meta_row.get("nature_eh")),
            "milieu_eh": _safe_str(meta_row.get("milieu_eh")),
        }

        piezo = df["niveau_nappe_eau"].dropna()
        piezo.name = "piezo"

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
    """Load continuous ERA5 data via the station-grid mapping table (single query)."""
    query = text("""
        SELECT e.era5_date AS date, e.total_precipitation,
               e.potential_evaporation, e.temperature_2m
        FROM gold.int_era5_for_all_stations e
        JOIN gold.int_station_era5_mapping m
          ON m.era5_latitude = e.latitude AND m.era5_longitude = e.longitude
        WHERE m.code_bss = :code_bss
        ORDER BY e.era5_date
    """)

    with engine.connect() as conn:
        era5_df = pd.read_sql(query, conn, params={"code_bss": code_bss}, parse_dates=["date"])

    if era5_df.empty:
        raise ValueError(f"No ERA5 data for station {code_bss}")

    era5_df = era5_df.set_index("date").sort_index()
    era5_df = era5_df[~era5_df.index.duplicated(keep="first")]

    precip = era5_df["total_precipitation"].clip(lower=0)
    evap = (-era5_df["potential_evaporation"]).clip(lower=0)
    temp = era5_df["temperature_2m"]

    for s in (precip, evap, temp):
        if s.index.freq is None:
            s = s.asfreq("D")

    return precip, evap, temp
