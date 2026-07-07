"""Station detail schema — mapped ERA5 cell coordinates (Task C2).

The station detail endpoints join the station→ERA5-cell mapping so the frontend
« Contexte climatique » section can query the climat point endpoints with the
cell's coordinates. Pure schema tests (repo convention: no DB touched).
"""
from api.schemas.observatory import HydroStation, PiezoStation


def test_piezo_station_exposes_mapped_era5_cell():
    s = PiezoStation(code_bss="07548X0009/F", era5_latitude=47.4, era5_longitude=0.7)
    assert s.era5_latitude == 47.4
    assert s.era5_longitude == 0.7


def test_piezo_station_era5_cell_defaults_to_none_for_unmapped_stations():
    s = PiezoStation(code_bss="07548X0009/F")
    assert s.era5_latitude is None
    assert s.era5_longitude is None


def test_hydro_station_exposes_mapped_era5_cell():
    s = HydroStation(code_station="K027401001", era5_latitude=45.0, era5_longitude=2.1)
    assert s.era5_latitude == 45.0
    assert s.era5_longitude == 2.1


def test_hydro_station_era5_cell_defaults_to_none_for_unmapped_stations():
    s = HydroStation(code_station="K027401001")
    assert s.era5_latitude is None
    assert s.era5_longitude is None
