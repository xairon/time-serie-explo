import inspect
from api.routers import observatory_common


def test_timeline_reads_fct_monthly_index_not_percent_rank():
    src = inspect.getsource(observatory_common.get_classification_timeline)
    assert "gold.fct_monthly_index" in src
    assert "PERCENT_RANK" not in src
