"""Pure CSV builder for single-station export (chronique + monthly index).

No FastAPI / no DB here — routers fetch the rows and call build_station_csv.

Format: tidy RFC-4180 CSV. Header on line 1, no comment block. Station identity
and provenance are denormalized into columns repeated on every row, so the file
is self-contained and machine-readable without out-of-band metadata. One CSV row
per day; the monthly standardized index (IPS for piezo, SSFI for hydro) of a
row's month is carried forward onto every day of that month.

Routers should encode the returned text as ``utf-8-sig`` (UTF-8 + BOM) so Excel
opens it without mojibake.
"""
from __future__ import annotations

import csv
import io
from datetime import date, datetime
from decimal import Decimal

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
_UNIT = {"piezo": "niveau en m NGF ; z-score sans unité",
         "hydro": "débit en m³/s ; z-score sans unité"}
_INDEX_REF = "1991-2020"
_SOURCE = "Junon / Hub'Eau + BRGM"

# Station identity columns, repeated on every row: (csv_header, meta_key)
_IDENTITY_COLS = [
    ("code", "code"),
    ("nom_station", "nom_commune"),
    ("code_departement", "code_departement"),
    ("nom_departement", "nom_departement"),
    ("codes_bdlisa", "codes_bdlisa"),
    ("latitude", "latitude"),
    ("longitude", "longitude"),
]
# Constant provenance columns appended at the end.
_PROVENANCE_HEADERS = ["index_ref", "unites", "source", "genere_le"]

# Canonical group keys, in CSV column order. `date` is always emitted and is
# not part of any group.
GROUP_KEYS = ("identity", "values", "meteo", "index", "provenance")


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
    if isinstance(v, Decimal):
        # Round to 6 dp, drop trailing zeros, avoid scientific notation.
        return format(round(v, 6).normalize(), "f")
    if isinstance(v, float):
        return repr(round(v, 6))
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


def build_station_csv(domain: str, meta: dict, daily_rows, index_rows, groups=None) -> str:
    if domain not in _INDEX_PREFIX:
        raise ValueError(f"unknown domain {domain!r}")
    active = set(GROUP_KEYS) if groups is None else (set(groups) & set(GROUP_KEYS))
    idx = index_by_month(index_rows)
    prefix = _INDEX_PREFIX[domain]
    value_cols = _VALUE_COLS[domain]

    identity_cells = [_fmt(meta.get(key)) for _, key in _IDENTITY_COLS]
    provenance_cells = [_INDEX_REF, _UNIT[domain], _SOURCE, _fmt(meta.get("generated_on"))]

    header = []
    if "identity" in active:
        header += [h for h, _ in _IDENTITY_COLS]
    header += ["date"]
    if "values" in active:
        header += [h for h, _ in value_cols]
    if "meteo" in active:
        header += [h for h, _ in _METEO_COLS]
    if "index" in active:
        header += ["mois_ref", f"{prefix}_z", f"{prefix}_classe", f"{prefix}_flag"]
    if "provenance" in active:
        header += _PROVENANCE_HEADERS

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(header)

    for row in daily_rows:
        mk = _month_key(row["date"])
        ix = idx.get(mk)
        rec = []
        if "identity" in active:
            rec += identity_cells
        rec.append(_fmt(row.get("date")))
        if "values" in active:
            rec += [_fmt(row.get(k)) for _, k in value_cols]
        if "meteo" in active:
            rec += [_fmt(row.get(k)) for _, k in _METEO_COLS]
        if "index" in active:
            if ix:
                rec += [f"{mk[0]:04d}-{mk[1]:02d}", _fmt(ix["z"]),
                        _fmt(ix["index_class"]), _fmt(ix["flag"])]
            else:
                rec += ["", "", "", ""]
        if "provenance" in active:
            rec += provenance_cells
        writer.writerow(rec)

    return out.getvalue()
