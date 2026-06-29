from datetime import date
from api.era5_anomaly import window_end_months, add_months, latest_complete_month


def test_window_end_months_no_wrap():
    assert window_end_months(3, 3) == [1, 2, 3]
    assert window_end_months(12, 1) == [12]
    assert window_end_months(6, 6) == [1, 2, 3, 4, 5, 6]


def test_window_end_months_wraps_year_boundary():
    assert window_end_months(1, 3) == [11, 12, 1]
    assert window_end_months(2, 12) == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2]


def test_add_months_forward_and_back():
    assert add_months(date(2024, 3, 15), 1) == date(2024, 4, 1)
    assert add_months(date(2024, 1, 10), -1) == date(2023, 12, 1)
    assert add_months(date(2024, 3, 1), -2) == date(2024, 1, 1)
    assert add_months(date(2024, 12, 1), 1) == date(2025, 1, 1)


def test_latest_complete_month_last_day():
    # March 31 = last day of March → March is complete
    assert latest_complete_month(date(2024, 3, 31)) == date(2024, 3, 1)


def test_latest_complete_month_mid_month():
    # March 15 → March incomplete → previous month = February
    assert latest_complete_month(date(2024, 3, 15)) == date(2024, 2, 1)


def test_latest_complete_month_jan_boundary():
    # Jan 1 (incomplete) → previous month = December of prior year
    assert latest_complete_month(date(2024, 1, 1)) == date(2023, 12, 1)
