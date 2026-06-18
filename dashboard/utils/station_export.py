"""Pure CSV builder for single-station export (chronique + monthly index).

No FastAPI / no DB here — routers fetch the rows and call build_station_csv.
One CSV row per day; the monthly standardized index (IPS for piezo, SSFI for
hydro) of a row's month is carried forward onto every day of that month.
"""
from __future__ import annotations

import csv
import io
from datetime import date, datetime

# Per-domain value columns: (csv_header, daily_row_key)
_VALUE_COLS = {
    "piezo": [("niveau_nappe_eau", "niveau_nappe_eau"),
              ("profondeur_nappe", "profondeur_nappe")],
    "hydro": [("resultat_obs_elab", "resultat_obs_elab"),
              ("grandeur_hydro_elab", "grandeur_hydro_elab")],
}
_METEO_COLS = [("temperature_2m", "temperature_2m"),
               ("total_precipitation", "total_precipitation"),
               ("potential_evaporation", "potential_evaporation")]
_INDEX_PREFIX = {"piezo": "ips", "hydro": "ssfi"}
_INDEX_LABEL = {"piezo": "IPS", "hydro": "SSFI"}
_UNIT = {"piezo": "niveau en m NGF", "hydro": "débit en m³/s"}


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.fromisoformat(str(d)[:10]).date()


def _month_key(d) -> tuple[int, int]:
    dd = _as_date(d)
    return (dd.year, dd.month)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (date, datetime)):
        return _as_date(v).isoformat()
    return str(v)


def index_by_month(index_rows) -> dict[tuple[int, int], dict]:
    """Map (year, month) -> {'z','index_class','flag'} from fct_monthly_index rows."""
    out: dict[tuple[int, int], dict] = {}
    for r in index_rows:
        out[_month_key(r["month"])] = {
            "z": r.get("z"),
            "index_class": r.get("index_class"),
            "flag": r.get("flag"),
        }
    return out


def _header_lines(domain: str, meta: dict, daily_rows, idx) -> list[str]:
    flag = ""
    if idx:
        flag = idx[max(idx.keys())].get("flag") or ""
    dmin = _fmt(daily_rows[0]["date"]) if daily_rows else ""
    dmax = _fmt(daily_rows[-1]["date"]) if daily_rows else ""
    return [
        f"Station: {meta.get('nom_commune') or ''} ({meta.get('code') or ''})",
        f"Département: {meta.get('nom_departement') or ''} ({meta.get('code_departement') or ''})",
        f"Coordonnées: {_fmt(meta.get('latitude'))}, {_fmt(meta.get('longitude'))}",
        f"Période exportée: {dmin} → {dmax}",
        f"Index: {_INDEX_LABEL[domain]} (réf. fixe 1991-2020, flag={flag})",
        f"Unités: {_UNIT[domain]} ; z-score sans unité",
        "Source: Junon / Hub'Eau + BRGM",
    ]


def build_station_csv(domain: str, meta: dict, daily_rows, index_rows) -> str:
    if domain not in _INDEX_PREFIX:
        raise ValueError(f"unknown domain {domain!r}")
    idx = index_by_month(index_rows)
    prefix = _INDEX_PREFIX[domain]
    value_cols = _VALUE_COLS[domain]

    out = io.StringIO()
    for line in _header_lines(domain, meta, daily_rows, idx):
        out.write(f"# {line}\n")

    writer = csv.writer(out)
    writer.writerow(
        ["date"]
        + [h for h, _ in value_cols]
        + [h for h, _ in _METEO_COLS]
        + ["mois_ref", f"{prefix}_z", f"{prefix}_classe", f"{prefix}_flag"]
    )

    for row in daily_rows:
        mk = _month_key(row["date"])
        ix = idx.get(mk)
        rec = [_fmt(row.get("date"))]
        rec += [_fmt(row.get(k)) for _, k in value_cols]
        rec += [_fmt(row.get(k)) for _, k in _METEO_COLS]
        if ix:
            rec += [f"{mk[0]:04d}-{mk[1]:02d}", _fmt(ix["z"]),
                    _fmt(ix["index_class"]), _fmt(ix["flag"])]
        else:
            rec += ["", "", "", ""]
        writer.writerow(rec)

    return out.getvalue()
