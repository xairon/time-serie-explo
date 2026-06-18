from datetime import date

from dashboard.utils.station_export import build_station_csv, index_by_month


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


def test_index_by_month_keys_on_year_month():
    idx = index_by_month(_piezo_index())
    assert idx[(2020, 1)] == {"z": -0.95, "index_class": "BAS", "flag": "normale"}


def test_csv_carries_index_forward_onto_each_day_of_month():
    csv_text = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index())
    lines = [l for l in csv_text.splitlines() if not l.startswith("#")]
    header = lines[0].split(",")
    assert header == ["date", "niveau_nappe_eau", "profondeur_nappe",
                      "temperature_2m", "total_precipitation", "potential_evaporation",
                      "mois_ref", "ips_z", "ips_classe", "ips_flag"]
    # both January days carry the same index value
    jan10 = lines[1].split(",")
    jan20 = lines[2].split(",")
    assert jan10[6:] == ["2020-01", "-0.95", "BAS", "normale"]
    assert jan20[6:] == ["2020-01", "-0.95", "BAS", "normale"]


def test_csv_leaves_index_cells_empty_for_month_without_index():
    csv_text = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index())
    data = [l for l in csv_text.splitlines() if not l.startswith("#")][1:]
    feb = data[2].split(",")  # 2020-02-05
    assert feb[0] == "2020-02-05"
    assert feb[6:] == ["", "", "", ""]


def test_header_block_contains_station_metadata():
    csv_text = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index())
    head = "\n".join(l for l in csv_text.splitlines() if l.startswith("#"))
    assert "BSS000/X" in head
    assert "Tours" in head
    assert "IPS" in head
    assert "1991-2020" in head
    assert "flag=normale" in head


def test_header_includes_generation_date_and_bdlisa_when_provided():
    meta = {**_meta(), "codes_bdlisa": "121AB01", "generated_on": "2026-06-18"}
    head = "\n".join(
        l for l in build_station_csv("piezo", meta, _piezo_daily(), _piezo_index()).splitlines()
        if l.startswith("#")
    )
    assert "généré le 2026-06-18" in head
    assert "BDLISA" in head and "121AB01" in head


def test_header_omits_optional_fields_when_absent():
    # _meta() has neither codes_bdlisa nor generated_on
    head = "\n".join(
        l for l in build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index()).splitlines()
        if l.startswith("#")
    )
    assert "BDLISA" not in head
    assert "généré le" not in head


def test_hydro_columns_and_unknown_domain():
    daily = [{"date": date(2020, 1, 10), "resultat_obs_elab": 1.7, "grandeur_hydro_elab": "Q",
              "temperature_2m": 5.0, "total_precipitation": 2.0, "potential_evaporation": 0.5}]
    index = [{"month": date(2020, 1, 1), "z": 2.3, "index_class": "EXTREMEMENT_HAUT", "flag": "adaptee"}]
    csv_text = build_station_csv("hydro", _meta(), daily, index)
    header = [l for l in csv_text.splitlines() if not l.startswith("#")][0].split(",")
    assert header == ["date", "resultat_obs_elab", "grandeur_hydro_elab",
                      "temperature_2m", "total_precipitation", "potential_evaporation",
                      "mois_ref", "ssfi_z", "ssfi_classe", "ssfi_flag"]
    import pytest
    with pytest.raises(ValueError):
        build_station_csv("nope", _meta(), [], [])
