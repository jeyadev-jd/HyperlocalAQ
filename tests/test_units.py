"""Tests for the WAQI unit conversion and the stale-station guard."""
import importlib.util
import math
import os
import pathlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "chennai area air quality.py"


@pytest.fixture(scope="module")
def hq():
    os.environ.setdefault("WAQI_API_TOKEN", "test")
    os.environ.setdefault("NASA_FIRMS_API_KEY", "test")
    spec = importlib.util.spec_from_file_location("hyperlocalaq", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def forward(hq, pollutant, native):
    """Independent forward EPA formula (native units -> AQI), used to check the inverse."""
    for c_lo, c_hi, i_lo, i_hi in hq.EPA_BREAKPOINTS[pollutant]:
        if native <= c_hi:
            return i_lo + (i_hi - i_lo) * (max(native, c_lo) - c_lo) / (c_hi - c_lo)
    return 500.0


def test_known_breakpoints(hq):
    c = hq.epa_aqi_to_concentration
    assert c("pm25", 100) == pytest.approx(35.4)
    assert c("pm25", 50) == pytest.approx(9.0)
    assert c("pm10", 100) == pytest.approx(154)
    assert c("no2", 50) == pytest.approx(53 * 46.0055 / 24.45)       # ppb to ug/m3
    assert c("o3", 100) == pytest.approx(0.070 * 1000 * 48.0 / 24.45)  # ppm to ug/m3
    assert c("co", 50) == pytest.approx(4.4 * 28.01 / 24.45)           # ppm to mg/m3


def test_the_original_bug_is_visible(hq):
    # A reported PM2.5 sub-index of 76 is about 22 ug/m3, not 76 ug/m3.
    assert hq.epa_aqi_to_concentration("pm25", 76) == pytest.approx(22.5, abs=0.2)
    assert hq.epa_aqi_to_concentration("pm25", 76) < 30


def test_monotonic_and_round_trip(hq):
    for p in ("pm25", "pm10", "no2", "so2", "o3", "co"):
        prev = -1.0
        for aqi in range(0, 501, 5):
            conc = hq.epa_aqi_to_concentration(p, aqi)
            assert conc >= prev, (p, aqi)
            prev = conc
        for aqi in (10, 40, 75, 120, 180, 250, 400):
            conc = hq.epa_aqi_to_concentration(p, aqi)
            native = conc
            if p in ("no2", "so2"):
                native = conc * 24.45 / hq._MOLAR_MASS[p]
            elif p == "o3":
                native = conc * 24.45 / hq._MOLAR_MASS[p] / 1000.0
            elif p == "co":
                native = conc * 24.45 / hq._MOLAR_MASS[p]
            assert forward(hq, p, native) == pytest.approx(aqi, abs=0.7), (p, aqi)


def test_invalid_inputs(hq):
    c = hq.epa_aqi_to_concentration
    assert math.isnan(c("pm25", np.nan))
    assert math.isnan(c("pm25", None))
    assert math.isnan(c("pm25", -5))
    assert math.isnan(c("voc", 50))
    assert c("pm25", 9999) == pytest.approx(500.4)  # capped at the top of the scale


def _fake_feed(iso, pm25=76):
    return {"status": "ok", "data": {"aqi": 76, "time": {"iso": iso},
            "city": {"name": "Test station", "geo": [13.0, 80.2]}, "iaqi": {"pm25": {"v": pm25}}}}


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def test_fetch_converts_fresh_station(hq, monkeypatch):
    now = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr(hq.requests, "get", lambda *a, **k: _Resp(_fake_feed(now)))
    df = hq.fetch_waqi_data(["x"], ["pm25"])
    assert len(df) == 1
    assert df.loc[0, "pm25_reported_aqi"] == 76
    assert df.loc[0, "pm25"] == pytest.approx(22.5, abs=0.2)


def test_stale_and_untimed_stations_are_ignored(hq, monkeypatch):
    old = (datetime.now(timezone.utc) - timedelta(days=118)).isoformat()
    monkeypatch.setattr(hq.requests, "get", lambda *a, **k: _Resp(_fake_feed(old)))
    assert hq.fetch_waqi_data(["x"], ["pm25"]).empty
    feed = _fake_feed("x"); feed["data"].pop("time")
    monkeypatch.setattr(hq.requests, "get", lambda *a, **k: _Resp(feed))
    assert hq.fetch_waqi_data(["x"], ["pm25"]).empty
