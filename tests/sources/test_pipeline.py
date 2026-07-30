import contextlib
import functools
import sqlite3

from datetime import datetime

import pandas as pd
import pytz

from eeweather.sources.engine import _fetch_year, observation_cache_key
from eeweather.sources.ghcnh import GHCNhSource
from eeweather.sources.pipeline import (
    align_to_range,
    data_gap_warnings,
    deserialize_hourly_data,
    load_year,
    read_cached_year,
    serialize_hourly_data,
)



GHCNH = GHCNhSource()
STATION = "USW00093134"
CACHE_KEY = observation_cache_key("ghcnh", STATION, 2007)


def _backdate_cache_key(store, key, updated):
    with contextlib.closing(sqlite3.connect(store._path)) as conn, conn:
        conn.execute(
            "update items set updated = ? where key = ?", (updated.isoformat(), key)
        )


def _load_2007(
    variables, read_from_cache=True, write_to_cache=True, fetch_from_web=True
):
    fetch = functools.partial(_fetch_year, GHCNH, STATION, STATION, 2007)
    block = load_year(
        CACHE_KEY, 2007, variables, fetch, GHCNH.cacheable,
        read_from_cache, write_to_cache, fetch_from_web,
    )

    return block


def _hourly_frame(start, end, value=1.0):
    index = pd.date_range(start, end, freq="h", tz="UTC")
    df = pd.DataFrame({"temperature": value}, index=index)

    return df


# serialization round-trips


def test_serialize_deserialize_hourly_data_round_trip(mock_api_transport):
    df = _fetch_year(GHCNH, STATION, STATION, 2007, ("temperature",))

    serialized = serialize_hourly_data(df)

    assert serialized["columns"] == ["temperature"]
    assert serialized["rows"][0][0] == "2007010100"
    assert len(serialized["rows"]) == len(df)

    round_tripped = deserialize_hourly_data(serialized)

    pd.testing.assert_frame_equal(round_tripped, df, check_freq=False)


def test_serialize_hourly_data_nan_round_trips_as_null(mock_api_transport):
    df = _fetch_year(GHCNH, "USW00093194", "USW00093194", 2013, ("temperature",))  # ends 2013-11-04

    serialized = serialize_hourly_data(df)

    assert any(row[1] is None for row in serialized["rows"])

    round_tripped = deserialize_hourly_data(serialized)

    pd.testing.assert_frame_equal(round_tripped, df, check_freq=False)


def test_serialize_multivariable_round_trip(mock_api_transport):
    df = _fetch_year(GHCNH, STATION, STATION, 2007, ("temperature", "wind_speed"))

    round_tripped = deserialize_hourly_data(serialize_hourly_data(df))

    pd.testing.assert_frame_equal(round_tripped, df, check_freq=False)


# cache freshness


def test_read_cached_year_empty(monkeypatch_key_value_store):
    assert read_cached_year(CACHE_KEY, 2007) is None


def test_read_cached_year_fresh(mock_api_transport, monkeypatch_key_value_store):
    _load_2007(("temperature",))

    assert read_cached_year(CACHE_KEY, 2007) is not None


def test_read_cached_year_expired_entry_is_cleared(
    mock_api_transport, monkeypatch_key_value_store
):
    _load_2007(("temperature",))

    # a cache entry written during its own data year goes stale
    _backdate_cache_key(
        monkeypatch_key_value_store, CACHE_KEY, pytz.UTC.localize(datetime(2007, 3, 3))
    )

    assert read_cached_year(CACHE_KEY, 2007) is None
    assert monkeypatch_key_value_store.key_exists(CACHE_KEY) is False


# per-year loads: cache reads, union refresh


def test_load_year_serves_from_cache(mock_api_transport, monkeypatch_key_value_store):
    df1 = _load_2007(("temperature",))
    df2 = _load_2007(("temperature",))

    pd.testing.assert_frame_equal(df1, df2, check_freq=False)


def test_load_year_variable_superset_refetches(
    mock_api_transport, monkeypatch_key_value_store
):
    df1 = _load_2007(("temperature",))
    assert list(df1.columns) == ["temperature"]

    # cache holds temperature only, so requesting more refetches the union
    df2 = _load_2007(("temperature", "wind_speed"))
    assert list(df2.columns) == ["temperature", "wind_speed"]

    # the refreshed cache entry now covers both variables
    cached = read_cached_year(CACHE_KEY, 2007)
    assert set(cached.columns) == {"temperature", "wind_speed"}

    # a temperature-only request serves the requested subset from cache
    df3 = _load_2007(("temperature",))
    assert list(df3.columns) == ["temperature"]


def test_load_year_refresh_keeps_cached_columns_the_request_omits(
    mock_api_transport, monkeypatch_key_value_store
):
    _load_2007(("temperature",))

    # wind_speed alone still fetches, and caches, the union with temperature
    df = _load_2007(("wind_speed",))
    assert list(df.columns) == ["wind_speed"]

    cached = read_cached_year(CACHE_KEY, 2007)
    assert set(cached.columns) == {"temperature", "wind_speed"}


def test_load_year_without_cache_or_web_returns_none(monkeypatch_key_value_store):
    assert _load_2007(("temperature",), fetch_from_web=False) is None


def test_load_year_without_write_leaves_the_cache_empty(
    mock_api_transport, monkeypatch_key_value_store
):
    df = _load_2007(("temperature",), write_to_cache=False)

    assert len(df) == 8760
    assert read_cached_year(CACHE_KEY, 2007) is None


# range alignment: the shared index every source path produces


def test_align_to_range_tick_ceils_start_and_floors_end():
    df = _hourly_frame("2020-01-01", "2020-01-02")
    offset = pd.tseries.frequencies.to_offset("h")

    aligned = align_to_range(
        df,
        pd.Timestamp("2020-01-01 00:20", tz="UTC"),
        pd.Timestamp("2020-01-01 05:40", tz="UTC"),
        offset,
    )

    assert aligned.index[0] == pd.Timestamp("2020-01-01 01:00", tz="UTC")
    assert aligned.index[-1] == pd.Timestamp("2020-01-01 05:00", tz="UTC")
    assert len(aligned) == 5


def test_align_to_range_non_tick_rolls_start_forward_and_end_back():
    df = _hourly_frame("2020-01-01", "2020-06-30")
    offset = pd.tseries.frequencies.to_offset("MS")

    aligned = align_to_range(
        df,
        pd.Timestamp("2020-01-15", tz="UTC"),
        pd.Timestamp("2020-05-20", tz="UTC"),
        offset,
    )

    assert aligned.index[0] == pd.Timestamp("2020-02-01", tz="UTC")
    assert aligned.index[-1] == pd.Timestamp("2020-05-01", tz="UTC")


def test_align_to_range_non_tick_start_inside_a_period_skips_that_period():
    # normalizing 2020-01-01 06:00 rolls forward onto a boundary before
    # start, so the partial first period is dropped
    df = _hourly_frame("2020-01-01", "2020-03-31")
    offset = pd.tseries.frequencies.to_offset("MS")

    aligned = align_to_range(
        df,
        pd.Timestamp("2020-01-01 06:00", tz="UTC"),
        pd.Timestamp("2020-03-31", tz="UTC"),
        offset,
    )

    assert aligned.index[0] == pd.Timestamp("2020-02-01", tz="UTC")
    assert aligned.index[-1] == pd.Timestamp("2020-03-01", tz="UTC")


def test_align_to_range_empty_frame_covers_the_whole_range():
    empty = pd.DataFrame(
        columns=["temperature"],
        index=pd.DatetimeIndex([], tz="UTC"),
        dtype=float,
    )
    offset = pd.tseries.frequencies.to_offset("h")

    aligned = align_to_range(
        empty,
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-01-02", tz="UTC"),
        offset,
    )

    assert len(aligned) == 25
    assert aligned.temperature.isna().all()


def test_align_to_range_gives_partial_frames_identical_indexes():
    start = pd.Timestamp("2020-01-01 00:30", tz="UTC")
    end = pd.Timestamp("2020-01-03 12:30", tz="UTC")
    offset = pd.tseries.frequencies.to_offset("h")
    early = _hourly_frame("2020-01-01", "2020-01-02")
    late = _hourly_frame("2020-01-02", "2020-01-05")

    aligned_early = align_to_range(early, start, end, offset)
    aligned_late = align_to_range(late, start, end, offset)

    assert aligned_early.index.equals(aligned_late.index)

    joined = pd.concat([aligned_early, aligned_late], axis=1)

    assert len(joined) == len(aligned_early)
    assert int(joined.iloc[:, 0].notna().sum()) == 24
    assert int(joined.iloc[:, 1].notna().sum()) == 37


# gap warnings


def test_data_gap_warnings_leading_gap():
    # synthetic edge case: a series whose data begins three days late
    index = pd.date_range(
        "2020-01-01", "2020-01-31", freq="h", tz="UTC"
    )
    ts = pd.Series(20.0, index=index)
    ts.iloc[: 24 * 3] = float("nan")

    warnings = data_gap_warnings(ts, "ghcnh", "temperature")

    assert [w.qualified_name for w in warnings] == ["eeweather.data_starts_late"]
    assert warnings[0].data["requested_start"] == "2020-01-01T00:00:00+00:00"
    assert warnings[0].data["first_valid"] == "2020-01-04T00:00:00+00:00"


def test_data_gap_warnings_empty_series_is_silent():
    ts = pd.Series([], dtype=float, index=pd.DatetimeIndex([], tz="UTC"))

    assert data_gap_warnings(ts, "ghcnh", "temperature") == []
