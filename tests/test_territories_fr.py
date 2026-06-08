from api.data.territories_fr import (
    DEPT_TO_REGION, REGION_NAMES, region_of, DEPARTMENT_NAMES,
)


def test_every_dept_maps_to_a_known_region():
    for dept, region_code in DEPT_TO_REGION.items():
        assert region_code in REGION_NAMES, f"{dept} -> unknown region {region_code}"


def test_region_of_known_and_unknown():
    assert region_of("45") == "24"          # Loiret -> Centre-Val de Loire
    assert region_of("75") == "11"          # Paris -> Île-de-France
    assert region_of("999") is None


def test_metropolitan_and_corsica_present():
    assert "2A" in DEPT_TO_REGION and "2B" in DEPT_TO_REGION
    assert "20" not in DEPT_TO_REGION
    assert len(DEPT_TO_REGION) == 101


def test_department_names_cover_all_mapped_depts():
    for dept in DEPT_TO_REGION:
        assert dept in DEPARTMENT_NAMES
