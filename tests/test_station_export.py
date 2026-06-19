from datetime import date
from decimal import Decimal

from dashboard.utils.station_export import GROUP_KEYS, build_station_csv, index_by_month

# Identity columns repeated on every row, in order.
_IDENTITY = ["code", "nom_station", "code_departement", "nom_departement",
             "codes_bdlisa", "latitude", "longitude"]
# Constant provenance columns appended at the end, in order.
_PROVENANCE = ["index_ref", "unites", "source", "genere_le"]


def _piezo_daily():
    # two days in Jan (has index), one day in Feb (no index)
    return [
        {"date": date(2020, 1, 10), "niveau_nappe_eau": 12.5, "profondeur_nappe": 3.1,
         "temperature_2m": 5.0, "total_precipitation": 2.0, "potential_evaporation": 0.5},
        {"date": date(2020, 1, 20), "niveau_nappe_eau": 12.7, "profondeur_nappe": 3.0,
         "temperature_2m": 6.0, "total_precipitation": 1.0, "potential_evaporation": 0.6},
        {"date": date(2020, 2, 5), "niveau_nappe_eau": 12.9, "profondeur_nappe": 2.9,
         "temperature_2m": 7.0, "total_precipitation": 0.0, "potential_evaporation": 0.7},
    ]


def _piezo_index():
    return [{"month": date(2020, 1, 1), "z": -0.95, "index_class": "BAS", "flag": "normale"}]


def _meta():
    return {"code": "BSS000/X", "nom_commune": "Tours", "code_departement": "37",
            "nom_departement": "Indre-et-Loire", "latitude": 47.39, "longitude": 0.69}


def _rows(csv_text):
    return [l for l in csv_text.splitlines() if l]


def test_index_by_month_keys_on_year_month():
    idx = index_by_month(_piezo_index())
    assert idx[(2020, 1)] == {"z": -0.95, "index_class": "BAS", "flag": "normale"}


def test_no_comment_block_header_on_first_line():
    csv_text = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index())
    # RFC-4180: no '#' metadata lines anywhere; header is line 1.
    assert "#" not in csv_text
    assert _rows(csv_text)[0].split(",")[0] == "code"


def test_piezo_full_column_order():
    csv_text = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index())
    header = _rows(csv_text)[0].split(",")
    assert header == (
        _IDENTITY
        + ["date", "niveau_nappe_eau", "profondeur_nappe",
           "temperature_2m", "total_precipitation", "potential_evaporation",
           "mois_ref", "ips_z", "ips_classe", "ips_flag"]
        + _PROVENANCE
    )


def test_identity_columns_repeated_on_every_data_row():
    csv_text = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index())
    for line in _rows(csv_text)[1:]:
        cells = line.split(",")
        assert cells[:7] == ["BSS000/X", "Tours", "37", "Indre-et-Loire", "", "47.39", "0.69"]


def test_provenance_columns_repeated_on_every_data_row():
    meta = {**_meta(), "generated_on": "2026-06-18"}
    csv_text = build_station_csv("piezo", meta, _piezo_daily(), _piezo_index())
    for line in _rows(csv_text)[1:]:
        cells = line.split(",")
        assert cells[-4:] == ["1991-2020", "niveau en m NGF ; z-score sans unité",
                              "Junon / Hub'Eau + BRGM", "2026-06-18"]


def test_csv_carries_index_forward_onto_each_day_of_month():
    csv_text = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index())
    rows = _rows(csv_text)
    # index columns sit just before the 4 provenance columns
    jan10 = rows[1].split(",")[-8:-4]
    jan20 = rows[2].split(",")[-8:-4]
    assert jan10 == ["2020-01", "-0.95", "BAS", "normale"]
    assert jan20 == ["2020-01", "-0.95", "BAS", "normale"]


def test_csv_leaves_index_cells_empty_for_month_without_index():
    csv_text = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index())
    feb = _rows(csv_text)[3].split(",")
    assert feb[7] == "2020-02-05"  # date column (after 7 identity cols)
    assert feb[-8:-4] == ["", "", "", ""]


def test_numbers_are_formatted_without_trailing_zeros():
    # NUMERIC from Postgres arrives as Decimal with long scale.
    # The warehouse delivers every numeric column as a high-precision Decimal,
    # not float — so rounding must apply to Decimal too, not just trailing-zero strip.
    daily = [{"date": date(2020, 1, 10),
              "niveau_nappe_eau": Decimal("172.8800000000000000"),
              "profondeur_nappe": Decimal("7.7700000000000000"),
              "temperature_2m": Decimal("17.078765869140625"),
              "total_precipitation": Decimal("0.0052474080584943295"),
              "potential_evaporation": None}]
    csv_text = build_station_csv("piezo", _meta(), daily, _piezo_index())
    cells = _rows(csv_text)[1].split(",")
    # date is at index 7 (after identity); values follow
    assert cells[8] == "172.88"
    assert cells[9] == "7.77"
    assert cells[10] == "17.078766"            # rounded to 6 dp
    assert cells[11] == "0.005247"             # rounded to 6 dp
    assert cells[12] == ""                     # None -> empty


def test_hydro_columns_and_unknown_domain():
    daily = [{"date": date(2020, 1, 10), "resultat_obs_elab": 1.7, "grandeur_hydro_elab": "Q",
              "temperature_2m": 5.0, "total_precipitation": 2.0, "potential_evaporation": 0.5}]
    index = [{"month": date(2020, 1, 1), "z": 2.3, "index_class": "EXTREMEMENT_HAUT", "flag": "adaptee"}]
    csv_text = build_station_csv("hydro", _meta(), daily, index)
    header = _rows(csv_text)[0].split(",")
    assert header == (
        _IDENTITY
        + ["date", "resultat_obs_elab", "grandeur_hydro_elab",
           "temperature_2m", "total_precipitation", "potential_evaporation",
           "mois_ref", "ssfi_z", "ssfi_classe", "ssfi_flag"]
        + _PROVENANCE
    )
    # hydro unit string in provenance
    assert _rows(csv_text)[1].split(",")[-3] == "débit en m³/s ; z-score sans unité"
    import pytest
    with pytest.raises(ValueError):
        build_station_csv("nope", _meta(), [], [])


def test_group_keys_canonical_order():
    assert GROUP_KEYS == ("identity", "values", "meteo", "index", "provenance")


def test_groups_subset_keeps_only_date_values_and_index():
    csv_text = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index(),
                                 groups={"values", "index"})
    header = _rows(csv_text)[0].split(",")
    assert header == ["date", "niveau_nappe_eau", "profondeur_nappe",
                      "mois_ref", "ips_z", "ips_classe", "ips_flag"]
    # a January data row carries date + values + index, nothing else
    jan10 = _rows(csv_text)[1].split(",")
    assert jan10 == ["2020-01-10", "12.5", "3.1", "2020-01", "-0.95", "BAS", "normale"]


def test_groups_none_is_unchanged_full_output():
    full = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index())
    explicit = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index(),
                                 groups=set(GROUP_KEYS))
    assert full == explicit


def test_groups_empty_emits_only_date():
    csv_text = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index(), groups=set())
    assert _rows(csv_text)[0].split(",") == ["date"]
    assert _rows(csv_text)[1].split(",") == ["2020-01-10"]


def test_groups_ignores_unknown_keys():
    csv_text = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index(),
                                 groups={"values", "bogus"})
    assert _rows(csv_text)[0].split(",") == ["date", "niveau_nappe_eau", "profondeur_nappe"]
