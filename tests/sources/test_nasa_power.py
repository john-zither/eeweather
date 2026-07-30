import gzip
import json

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import requests

from eeweather.sources.nasa_power import NASAPowerSource
from eeweather.sources.nasa_power.source import (
    API_REQUEST_TRIES,
    MAX_PARAMETERS,
    MET_PARAMETERS,
    SOLAR_PARAMETERS,
    _parse,
)



FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"

MET_FIXTURE = "nasa_power_met_34.02_-118.29_2024.json.gz"

SOLAR_FIXTURE = "nasa_power_solar_34.02_-118.29_2024.json.gz"

MET_NATIVE = tuple(parameter.native for parameter in MET_PARAMETERS)

SOLAR_NATIVE = tuple(parameter.native for parameter in SOLAR_PARAMETERS)

POINT = (34.02, -118.29)

START = date(2024, 6, 1)

END = date(2024, 6, 30)

# a clear June afternoon hour and a June night hour at the fixture point
NOON = pd.Timestamp("2024-06-15 20:00", tz="UTC")

NIGHT = pd.Timestamp("2024-06-15 10:00", tz="UTC")

_ERROR_BODY = json.loads(
    (FIXTURE_DIR / "nasa_power_error_too_many_parameters.json").read_text()
)


def _fixture_payload(name):
    with gzip.open(FIXTURE_DIR / name, "rb") as f:
        payload = json.loads(f.read().decode())

    return payload


class MockResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        if self.payload is None:
            raise ValueError("no json body")

        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                "{} error".format(self.status_code), response=self
            )


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr("eeweather.sources.nasa_power.source.time.sleep", sleeps.append)

    return sleeps


@pytest.fixture
def responses(monkeypatch):
    """Serve a scripted response per attempt, recording each request."""
    scripted = []
    calls = []

    def mock_get(url, params=None):
        calls.append(params)

        return scripted[min(len(calls) - 1, len(scripted) - 1)]

    monkeypatch.setattr("eeweather.sources.nasa_power.source._get", mock_get)

    return scripted, calls


@pytest.fixture
def chunked_transport(monkeypatch):
    """Serve both captured families from one point-year as a single grid.

    The two fixtures cover the same point and range, so a request for
    any subset of the 30 parameters is served by slicing them; the
    fixture records the parameter list of each submission.
    """
    payload = _fixture_payload(MET_FIXTURE)
    solar = _fixture_payload(SOLAR_FIXTURE)
    payload["parameters"].update(solar["parameters"])
    payload["properties"]["parameter"].update(solar["properties"]["parameter"])
    requested = []

    def mock_get(url, params=None):
        names = params["parameters"].split(",")
        requested.append(names)
        served = dict(payload)
        served["parameters"] = {name: payload["parameters"][name] for name in names}
        served["properties"] = {
            "parameter": {
                name: payload["properties"]["parameter"][name] for name in names
            }
        }

        return MockResponse(served)

    monkeypatch.setattr("eeweather.sources.nasa_power.source._get", mock_get)

    return requested


def _fetch(parameters):
    source = NASAPowerSource()
    block = source.fetch(POINT[0], POINT[1], START, END, parameters)

    return block


def test_fetch_pins_the_submission_parameters(responses):
    scripted, calls = responses
    scripted.append(MockResponse(_fixture_payload(MET_FIXTURE)))

    _fetch(("T2M", "PS"))

    assert len(calls) == 1
    assert calls[0] == {
        "parameters": "T2M,PS",
        "community": "RE",
        "latitude": 34.02,
        "longitude": -118.29,
        "start": "20240601",
        "end": "20240630",
        "format": "JSON",
        "time-standard": "UTC",
    }


def test_fetch_indexes_hourly_values_in_utc(mock_nasa_power_transport):
    block = _fetch(MET_NATIVE)

    assert isinstance(block.data.index, pd.DatetimeIndex)
    assert str(block.data.index.tz) == "UTC"
    assert block.data.index[0] == pd.Timestamp("2024-06-01 00:00", tz="UTC")
    assert block.data.index[-1] == pd.Timestamp("2024-06-30 23:00", tz="UTC")
    assert len(block.data) == 720


def test_fetch_renames_met_parameters_to_canonical_columns(mock_nasa_power_transport):
    block = _fetch(MET_NATIVE)

    assert tuple(block.data.columns) == (
        "temperature",
        "dew_point_temperature",
        "relative_humidity",
        "wind_speed",
        "specific_humidity",
        "skin_temperature",
        "soil_temperature",
        "eastward_wind",
        "northward_wind",
        "surface_roughness",
        "surface_pressure",
        "precipitation",
        "snowfall",
        "snow_cover",
    )
    assert block.data.loc[NOON, "temperature"] == 30.11


def test_fetch_renames_solar_parameters_to_canonical_columns(mock_nasa_power_transport):
    block = _fetch(SOLAR_NATIVE)

    assert tuple(block.data.columns) == (
        "ghi",
        "clearsky_ghi",
        "dni",
        "clearsky_dni",
        "dhi",
        "clearsky_dhi",
        "bhi",
        "clearsky_bhi",
        "albedo",
        "longwave_down",
        "longwave_up",
        "airmass",
        "aerosol_optical_depth_550",
        "aerosol_optical_depth_840",
        "precipitable_water",
        "cloud_cover",
    )


def test_diffuse_and_direct_map_to_the_original_pair(mock_nasa_power_transport):
    block = _fetch(("ORIGINAL_ALLSKY_SFC_SW_DIFF", "ORIGINAL_ALLSKY_SFC_SW_DIRH"))

    assert tuple(block.data.columns) == ("dhi", "bhi")
    assert block.data.loc[NOON, "dhi"] == 134.05
    assert block.data.loc[NOON, "bhi"] == 876.20


def test_original_components_sum_to_ghi(mock_nasa_power_transport):
    block = _fetch(SOLAR_NATIVE)
    hour = block.data.loc[NOON]

    assert hour["ghi"] == 1010.25
    assert hour["dhi"] + hour["bhi"] == pytest.approx(1010.25, abs=1e-9)


def test_night_fill_becomes_nan_and_night_zero_irradiance_survives(
    mock_nasa_power_transport,
):
    block = _fetch(SOLAR_NATIVE)
    hour = block.data.loc[NIGHT]

    # genuinely zero at night, not fill
    assert hour["ghi"] == 0.0
    assert hour["dni"] == 0.0
    # undefined at night, served as the header's fill value
    assert pd.isna(hour["albedo"])
    assert pd.isna(hour["airmass"])
    # the same fields are defined by day
    assert block.data.loc[NOON, "albedo"] == 0.16
    assert block.data.loc[NOON, "airmass"] == 1.03


def test_surface_pressure_converts_kpa_to_hpa(mock_nasa_power_transport):
    block = _fetch(("PS",))

    assert block.data.loc[NOON, "surface_pressure"] == pytest.approx(963.8, abs=1e-9)


def test_precipitation_rate_converts_to_hourly_depth(mock_nasa_power_transport):
    block = _fetch(("PRECTOTCORR",))

    # 0.89 mm/day over the hour beginning 2024-06-02 12Z
    assert block.data.loc[
        pd.Timestamp("2024-06-02 12:00", tz="UTC"), "precipitation"
    ] == pytest.approx(0.0370833333, abs=1e-9)


def test_fetch_reports_the_response_provenance(mock_nasa_power_transport):
    met = _fetch(MET_NATIVE)
    solar = _fetch(SOLAR_NATIVE)

    assert met.sources == ("MERRA2", "POWER")
    assert solar.sources == ("SYN1DEG", "POWER")
    assert met.api_version == "v2.9.6"


def test_a_family_fits_in_one_submission(chunked_transport):
    _fetch(SOLAR_NATIVE)

    assert len(SOLAR_NATIVE) <= MAX_PARAMETERS
    assert chunked_transport == [list(SOLAR_NATIVE)]


def test_fetch_chunks_parameters_beyond_the_api_limit(chunked_transport):
    requested = chunked_transport
    parameters = MET_NATIVE + SOLAR_NATIVE

    block = _fetch(parameters)

    assert len(parameters) == 30
    assert [len(chunk) for chunk in requested] == [20, 10]
    assert requested[0] + requested[1] == list(parameters)
    assert tuple(block.data.columns) == tuple(
        parameter.canonical for parameter in MET_PARAMETERS + SOLAR_PARAMETERS
    )
    assert block.data.loc[NOON, "temperature"] == 30.11
    assert block.data.loc[NOON, "ghi"] == 1010.25
    assert len(block.data) == 720


def test_rate_limit_is_retried(responses, no_sleep):
    scripted, calls = responses
    scripted.append(MockResponse(None, status_code=429))
    scripted.append(MockResponse(_fixture_payload(MET_FIXTURE)))

    block = _fetch(("T2M",))

    assert len(calls) == 2
    assert len(no_sleep) == 1
    assert block.data.loc[NOON, "temperature"] == 30.11


def test_server_error_is_retried_until_the_attempts_run_out(responses, no_sleep):
    scripted, calls = responses
    scripted.append(MockResponse(None, status_code=503))

    with pytest.raises(requests.HTTPError):
        _fetch(("T2M",))

    assert len(calls) == API_REQUEST_TRIES
    assert len(no_sleep) == API_REQUEST_TRIES - 1
    # exponential backoff with jitter: each wait falls in its own window
    assert 2 <= no_sleep[0] < 4
    assert 4 <= no_sleep[1] < 8
    assert 8 <= no_sleep[2] < 16


def test_client_error_is_not_retried(responses, no_sleep):
    scripted, calls = responses
    scripted.append(MockResponse(None, status_code=404))

    with pytest.raises(requests.HTTPError):
        _fetch(("T2M",))

    assert len(calls) == 1
    assert no_sleep == []


def test_retry_after_is_honored(responses, no_sleep):
    scripted, calls = responses
    scripted.append(
        MockResponse(None, status_code=429, headers={"Retry-After": "7"})
    )
    scripted.append(MockResponse(_fixture_payload(MET_FIXTURE)))

    _fetch(("T2M",))

    assert no_sleep == [7.0]


def test_rejected_submission_raises_with_the_api_messages(responses):
    scripted, calls = responses
    scripted.append(MockResponse(_ERROR_BODY, status_code=422))

    with pytest.raises(ValueError) as excinfo:
        _fetch(("T2M",))

    assert "maximum of 20 parameters" in str(excinfo.value)
    assert len(calls) == 1


def test_error_document_served_with_a_success_status_raises(responses):
    scripted, calls = responses
    scripted.append(MockResponse(_ERROR_BODY))

    with pytest.raises(ValueError) as excinfo:
        _fetch(("T2M",))

    assert "maximum of 20 parameters" in str(excinfo.value)


def test_unexpected_response_unit_raises(mock_nasa_power_transport):
    payload = _fixture_payload(MET_FIXTURE)
    payload["parameters"]["PS"]["units"] = "hPa"

    with pytest.raises(ValueError) as excinfo:
        _parse(payload, ("PS",))

    assert "PS" in str(excinfo.value)
    assert "hPa" in str(excinfo.value)


def test_transport_fixture_refuses_unknown_requests(mock_nasa_power_transport):
    source = NASAPowerSource()

    with pytest.raises(AssertionError):
        source.fetch(40.0, -80.0, START, END, ("T2M",))
