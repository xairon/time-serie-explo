"""Tests for Pastas model builder."""
from __future__ import annotations

import pytest
import pastas as ps

from dashboard.utils.pastas.builder import build_model, ValidationError


def test_build_model_gamma_linear(synthetic_station):
    model, tmin, tmax = build_model(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="Linear",
        response_type="Gamma",
        noise_type="ArNoiseModel",
        tmin=None,
        tmax=None,
    )
    assert isinstance(model, ps.Model)
    assert len(model.stressmodels) == 1
    assert model.noisemodel is not None


def test_build_model_exponential_flexmodel(synthetic_station):
    model, _, _ = build_model(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="FlexModel",
        response_type="Exponential",
        noise_type="none",
        tmin=None,
        tmax=None,
    )
    assert isinstance(model, ps.Model)
    assert model.noisemodel is None


def test_build_model_hantush(synthetic_station):
    model, _, _ = build_model(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="Linear",
        response_type="Hantush",
        noise_type="ArNoiseModel",
        tmin=None,
        tmax=None,
    )
    assert isinstance(model, ps.Model)


def test_build_model_custom_window(synthetic_station):
    model, tmin, tmax = build_model(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="Linear",
        response_type="Gamma",
        noise_type="ArNoiseModel",
        tmin="2016-01-01",
        tmax="2018-12-31",
    )
    assert str(tmin) == "2016-01-01"
    assert str(tmax) == "2018-12-31"


def test_build_model_rejects_short_series():
    import pandas as pd
    import numpy as np
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    short = pd.Series(np.zeros(100), index=dates)

    with pytest.raises(ValidationError, match="at least 365"):
        build_model(
            gwl=short, precip=short, evap=short,
            recharge_type="Linear", response_type="Gamma",
            noise_type="none", tmin=None, tmax=None,
        )


def test_build_model_rejects_high_nan_ratio():
    import pandas as pd
    import numpy as np
    dates = pd.date_range("2015-01-01", periods=500, freq="D")
    gwl = pd.Series(np.ones(500), index=dates)
    gwl.iloc[:150] = np.nan

    precip = pd.Series(np.ones(500), index=dates)
    evap = pd.Series(np.ones(500), index=dates)

    with pytest.raises(ValidationError, match="NaN"):
        build_model(
            gwl=gwl, precip=precip, evap=evap,
            recharge_type="Linear", response_type="Gamma",
            noise_type="none", tmin=None, tmax=None,
        )


def test_build_model_rejects_no_overlap():
    import pandas as pd
    import numpy as np
    dates_gwl = pd.date_range("2015-01-01", periods=500, freq="D")
    dates_stress = pd.date_range("2020-01-01", periods=500, freq="D")

    gwl = pd.Series(np.ones(500), index=dates_gwl)
    precip = pd.Series(np.ones(500), index=dates_stress)
    evap = pd.Series(np.ones(500), index=dates_stress)

    with pytest.raises(ValidationError, match="overlap"):
        build_model(
            gwl=gwl, precip=precip, evap=evap,
            recharge_type="Linear", response_type="Gamma",
            noise_type="none", tmin=None, tmax=None,
        )
