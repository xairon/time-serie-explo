"""Pydantic models for the Observatory module.

Covers piezo, hydro, ERA5, common (alerts, stats), and Pastas observatory types.
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Piezo
# ---------------------------------------------------------------------------

class PiezoStation(BaseModel):
    code_bss: str
    bss_id: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    nom_commune: str | None = None
    code_departement: str | None = None
    nom_departement: str | None = None
    codes_bdlisa: str | None = None
    altitude_station: float | None = None
    date_debut_mesure: date | None = None
    date_fin_mesure: date | None = None
    nb_mesures_total: float | None = None
    nb_mois_total: int | None = None
    premiere_mesure: date | None = None
    derniere_mesure: date | None = None
    niveau_moyen_global: float | None = None
    niveau_min_absolu: float | None = None
    niveau_max_absolu: float | None = None
    niveau_stddev_global: float | None = None
    amplitude_totale: float | None = None
    profondeur_moyenne_globale: float | None = None
    temperature_moyenne_globale: float | None = None
    precipitation_moyenne_mensuelle: float | None = None
    derniere_annee: int | None = None
    niveau_derniere_annee: float | None = None
    classification_derniere_annee: str | None = None
    percentile_derniere_annee: float | None = None
    slope_niveau: float | None = None
    r2_niveau: float | None = None
    slope_precipitation: float | None = None
    nb_mois_tendance: int | None = None
    tendance_classification: str | None = None
    niveau_alerte: str | None = None
    qualite_tendance: str | None = None


class PiezoDaily(BaseModel):
    date: date
    niveau_nappe_eau: float | None = None
    profondeur_nappe: float | None = None
    temperature_2m: float | None = None
    total_precipitation: float | None = None
    potential_evaporation: float | None = None


class PiezoMonthly(BaseModel):
    mois: date
    niveau_moyen: float | None = None
    niveau_min: float | None = None
    niveau_max: float | None = None
    amplitude_mensuelle: float | None = None
    temperature_moyenne: float | None = None
    precipitation_totale: float | None = None
    evaporation_moyenne: float | None = None
    nb_jours_mesures: int | None = None
    niveau_moy_mobile_3m: float | None = None
    niveau_moy_mobile_12m: float | None = None
    precipitation_moy_mobile_12m: float | None = None
    variation_niveau_vs_mois_prec: float | None = None
    variation_niveau_vs_annee_prec: float | None = None


class PiezoYearly(BaseModel):
    annee: int
    niveau_moyen_annuel: float | None = None
    niveau_min_annuel: float | None = None
    niveau_max_annuel: float | None = None
    amplitude_annuelle: float | None = None
    temperature_moyenne_annuelle: float | None = None
    precipitation_totale_annuelle: float | None = None
    bilan_hydrique_annuel: float | None = None
    nb_jours_mesures_annuel: float | None = None
    percentile_niveau_historique: float | None = None
    classification_niveau_annuel: str | None = None
    niveau_moy_mobile_5ans: float | None = None


class PiezoTrend(BaseModel):
    code_bss: str
    saison: str | None = None
    code_departement: str | None = None
    nom_departement: str | None = None
    variation_annuelle_m: float | None = None
    fiabilite_tendance: float | None = None
    nb_points: int | None = None
    classification_tendance: str | None = None
    projection_variation_5ans_m: float | None = None


class PiezoPercentiles(BaseModel):
    p10: float | None = None
    p25: float | None = None
    p75: float | None = None
    p90: float | None = None


class PiezoSPLI(BaseModel):
    mois: date
    value: float | None = None
    spli: float | None = None
    classification: str = "UNKNOWN"


class PiezoSPI(BaseModel):
    mois: date
    value: float | None = None
    spi: float | None = None
    classification: str = "UNKNOWN"


class PiezoSiblingStation(BaseModel):
    code_bss: str
    nom_commune: str | None = None
    code_departement: str | None = None
    classification: str | None = None
    derniere_mesure: date | None = None
    distance_km: float | None = None


class PiezoBasinSiblings(BaseModel):
    code_bdlisa: str
    nom_bdlisa: str | None = None
    nature_bdlisa: str | None = None
    nb_stations: int
    siblings: list[PiezoSiblingStation]


# ---------------------------------------------------------------------------
# Hydro
# ---------------------------------------------------------------------------

class HydroStation(BaseModel):
    code_station: str
    libelle_station: str | None = None
    code_site: str | None = None
    libelle_site: str | None = None
    code_cours_eau: str | None = None
    nom_cours_eau: str | None = None
    code_departement: str | None = None
    nom_departement: str | None = None
    date_ouverture_station: date | None = None
    longitude_station: float | None = None
    latitude_station: float | None = None
    grandeur_hydro_principale: str | None = None
    premiere_mesure: date | None = None
    derniere_mesure: date | None = None
    nb_jours_total: float | None = None
    nb_mois_total: int | None = None
    resultat_moyen_global: float | None = None
    resultat_min_global: float | None = None
    resultat_max_global: float | None = None
    resultat_stddev_global: float | None = None
    annee_dernier_bilan: int | None = None
    resultat_moyen_dern_annee: float | None = None
    classification_resultat_dern_annee: str | None = None
    percentile_resultat_dern_annee: float | None = None


class HydroDaily(BaseModel):
    date: date
    resultat_obs_elab: float | None = None
    grandeur_hydro_elab: str | None = None
    temperature_2m: float | None = None
    total_precipitation: float | None = None
    potential_evaporation: float | None = None


class HydroMonthly(BaseModel):
    mois: date
    resultat_moyen: float | None = None
    resultat_min: float | None = None
    resultat_max: float | None = None
    amplitude_mensuelle: float | None = None
    temperature_moyenne: float | None = None
    precipitation_totale: float | None = None
    evaporation_moyenne: float | None = None
    nb_jours_mesures: int | None = None
    resultat_moy_mobile_3m: float | None = None
    resultat_moy_mobile_12m: float | None = None
    precipitation_moy_mobile_12m: float | None = None
    variation_resultat_vs_mois_prec: float | None = None
    variation_resultat_vs_annee_prec: float | None = None


class HydroYearly(BaseModel):
    annee: int
    resultat_moyen_annuel: float | None = None
    resultat_min_annuel: float | None = None
    resultat_max_annuel: float | None = None
    amplitude_annuelle: float | None = None
    temperature_moyenne_annuelle: float | None = None
    precipitation_totale_annuelle: float | None = None
    nb_jours_mesures_annuel: float | None = None
    percentile_resultat_historique: float | None = None
    classification_resultat_annuel: str | None = None


class HydroTrend(BaseModel):
    code_station: str
    grandeur_hydro_elab: str | None = None
    saison: str | None = None
    code_departement: str | None = None
    nom_departement: str | None = None
    variation_annuelle: float | None = None
    fiabilite_tendance: float | None = None
    nb_points: int | None = None
    classification_tendance: str | None = None
    projection_variation_5ans: float | None = None


class HydroPercentiles(BaseModel):
    p10: float | None = None
    p25: float | None = None
    p75: float | None = None
    p90: float | None = None


class HydroSSFI(BaseModel):
    mois: date
    value: float | None = None
    ssfi: float | None = None
    classification: str = "UNKNOWN"


class HydroSPI(BaseModel):
    mois: date
    value: float | None = None
    spi: float | None = None
    classification: str = "UNKNOWN"


class HydroSiblingStation(BaseModel):
    code_station: str
    libelle_station: str | None = None
    grandeur_hydro_principale: str | None = None
    classification: str | None = None
    derniere_mesure: date | None = None


class HydroSiteSiblings(BaseModel):
    code_site: str
    libelle_site: str | None = None
    nom_cours_eau: str | None = None
    nb_stations: int
    siblings: list[HydroSiblingStation]


# ---------------------------------------------------------------------------
# Common (alerts, stats)
# ---------------------------------------------------------------------------

class Alert(BaseModel):
    code: str
    type: str
    latitude: float | None = None
    longitude: float | None = None
    commune: str | None = None
    code_departement: str | None = None
    departement: str | None = None
    classification: str | None = None
    derniere_mesure: str | None = None
    alerte_depuis_annee: int | None = None
    nb_annees_consecutives: int | None = None


class NationalStats(BaseModel):
    total_piezo: int
    piezo_extremement_bas: int = 0
    piezo_tres_bas: int = 0
    piezo_bas: int = 0
    piezo_normal: int = 0
    piezo_haut: int = 0
    piezo_tres_haut: int = 0
    piezo_extremement_haut: int = 0
    piezo_no_class: int = 0
    total_hydro: int
    hydro_extremement_bas: int = 0
    hydro_tres_bas: int = 0
    hydro_bas: int = 0
    hydro_normal: int = 0
    hydro_haut: int = 0
    hydro_tres_haut: int = 0
    hydro_extremement_haut: int = 0


class DepartmentStats(BaseModel):
    code_departement: str
    nom_departement: str | None = None
    nb_piezo: int
    nb_hydro: int
    pct_tres_bas: float | None = None
    avg_variation: float | None = None


# ---------------------------------------------------------------------------
# ERA5
# ---------------------------------------------------------------------------

class ERA5GridPoint(BaseModel):
    era5_latitude: float
    era5_longitude: float


class ERA5Snapshot(BaseModel):
    latitude: float
    longitude: float
    temperature_2m: float | None = None
    total_precipitation: float | None = None
    potential_evaporation: float | None = None


# ---------------------------------------------------------------------------
# Pastas Observatory (from BRGM warehouse pre-computed tables)
# ---------------------------------------------------------------------------

class PastasSummary(BaseModel):
    code_bss: str
    evp: float | None = None
    nash: float | None = None
    kge: float | None = None
    rmse: float | None = None
    r2: float | None = None
    tmax_days: float | None = None
    cutoff_95_days: float | None = None
    gain: float | None = None
    mean_response_time: float | None = None
    block_response: list[float] | None = None
    autocorr_time: float | None = None
    recession_constant: float | None = None
    recovery_constant: float | None = None
    parde_seasonality: float | None = None
    avg_seasonal_fluctuation: float | None = None
    colwell_constancy: float | None = None
    duration_curve_slope: float | None = None
    baselevel_index: float | None = None
    series_start: date | None = None
    series_end: date | None = None
    series_length_days: int | None = None
    n_observations: int | None = None
    fitted_at: date | None = None
    pastas_version: str | None = None


class PastasTimeseriesPoint(BaseModel):
    date: date
    simulated: float | None = None
    observed: float | None = None
    residuals: float | None = None
    recharge_contribution: float | None = None
    wb_recharge: float | None = None
    wb_actual_evaporation: float | None = None
    wb_surface_runoff: float | None = None
    wb_effective_precip: float | None = None


class PastasSGIPoint(BaseModel):
    date: date
    sgi: float | None = None
    classification: str = "UNKNOWN"


class PastasCoverage(BaseModel):
    code_bss: str
    evp: float | None = None
    nash: float | None = None
    tmax_days: float | None = None


# ---------------------------------------------------------------------------
# GeoJSON feature (used by common router, not a full model — just for type hints)
# ---------------------------------------------------------------------------

class StationGeoJSONFeature(BaseModel):
    """Lightweight GeoJSON Feature for a station marker."""
    type: str = "Feature"
    geometry: dict
    properties: dict
