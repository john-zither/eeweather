"""Shared load machinery, independent of how a source is keyed.

Station feeds and gridded sources alike cache one hourly block per year,
refresh it by variable union, align it to the requested range, aggregate
it to the requested frequency, and report coverage gaps through this
module. Because every path aligns here, frames for one request range and
frequency share an identical UTC index and join safely.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

import eeweather.cache
from ..exceptions import EEWeatherWarning
from .vocabulary import aggregation_for



TRAILING_GAP_WARNING_THRESHOLD = timedelta(days=1)
LEADING_GAP_WARNING_THRESHOLD = timedelta(days=1)
INTERNAL_GAP_WARNING_THRESHOLD = timedelta(days=7)


def store():
    return eeweather.cache.key_value_store_proxy.get_store()


def serialize_hourly_data(df):
    rows = [
        [index.strftime("%Y%m%d%H")] + values
        for index, values in zip(
            df.index, df.astype(object).where(df.notna(), None).values.tolist()
        )
    ]
    serialized = {"columns": list(df.columns), "rows": rows}

    return serialized


def deserialize_hourly_data(data):
    index = pd.to_datetime(
        [row[0] for row in data["rows"]], format="%Y%m%d%H", utc=True
    )
    df = pd.DataFrame(
        [row[1:] for row in data["rows"]],
        index=index,
        columns=data["columns"],
        dtype=float,
    )

    return df.sort_index().resample("h").mean()


def read_cached_year(key, year):
    """The fresh cached block under a key, or None. An entry too old for
    its data year is dropped."""
    cache = store()
    if not cache.key_exists(key):
        return None
    if eeweather.cache._expired(cache.key_updated(key), year):
        cache.clear(key)

        return None

    return deserialize_hourly_data(cache.retrieve_json(key))


def load_year(
    key, year, variables, fetch, cacheable,
    read_from_cache, write_to_cache, fetch_from_web,
):
    """One year of hourly data under a cache key, from cache when it
    covers the request.

    A cache entry serves the request when it is fresh and holds every
    requested variable. Otherwise ``fetch`` is called with the union of
    the requested and already-cached variables, so a refresh never drops
    a column, and its frame is returned reindexed to the requested
    columns. Returns None when only a fetch could serve the request and
    fetching is disabled.
    """
    cached = None
    if cacheable:
        cached = read_cached_year(key, year)

    cache_covers_request = cached is not None and set(variables) <= set(cached.columns)
    if read_from_cache and cache_covers_request:
        return cached[list(variables)]

    if not fetch_from_web:
        return None

    if cached is None:
        cached_columns = ()
    else:
        cached_columns = tuple(cached.columns)
    fetch_variables = tuple(dict.fromkeys(variables + cached_columns))
    df = fetch(fetch_variables)
    if cacheable and write_to_cache:
        store().save_json(key, serialize_hourly_data(df))

    return df.reindex(columns=list(variables))


def align_to_range(df, start, end, offset):
    """The frame on every period of the requested range at one frequency.

    Start and end fall exactly on period boundaries: a period partially
    before start is excluded, and end's own period is included. The full
    range is covered even when no data loaded, so periods without data
    are NaN and two frames aligned over the same range and frequency
    share an index.
    """
    df = df[start:end]
    if isinstance(offset, pd.tseries.offsets.Tick):
        range_start = pd.Timestamp(start).ceil(offset)
        range_end = pd.Timestamp(end).floor(offset)
    else:
        range_start = offset.rollforward(pd.Timestamp(start).normalize())
        if range_start < pd.Timestamp(start):
            range_start = range_start + offset
        range_end = offset.rollback(pd.Timestamp(end).normalize())

    return df.reindex(pd.date_range(range_start, range_end, freq=offset))


def resample_by_vocabulary(df, offset):
    """Hourly values at the requested frequency, column by column.

    Coarser periods roll up by the column's vocabulary aggregation; a
    period with no data at all is NaN regardless of aggregation. Every
    label is its period's start. Sub-hourly slots interpolate
    point-in-time columns linearly between hourly values and spread
    accumulations evenly, never crossing a missing hour.
    """
    if isinstance(offset, pd.tseries.offsets.Tick) and (
        pd.Timedelta(offset) < pd.Timedelta(hours=1)
    ):
        return _upsample(df, offset)

    resampled = df.resample(offset, label="left", closed="left")
    columns = {}
    for column in df.columns:
        aggregation = aggregation_for(column)
        if aggregation == "sum":
            columns[column] = resampled[column].sum(min_count=1)
        else:
            columns[column] = getattr(resampled[column], aggregation)()
    aggregated = pd.DataFrame(columns)

    return aggregated


def _upsample(df, offset):
    """Hourly values at a finer frequency; the offset must divide the
    hour evenly."""
    step = pd.Timedelta(offset)
    if pd.Timedelta(hours=1) % step != pd.Timedelta(0):
        raise ValueError(
            "A sub-hourly frequency must divide the hour evenly,"
            " got: {}".format(offset.freqstr)
        )
    slots = int(pd.Timedelta(hours=1) / step)

    up = df.resample(offset).asfreq()
    columns = {}
    for column in df.columns:
        if aggregation_for(column) == "sum":
            spread = df[column].reindex(up.index, method="ffill", limit=slots - 1)
            columns[column] = spread / slots
        else:
            filled = up[column].interpolate(method="linear", limit_area="inside")
            valid = df[column].notna()
            left_valid = valid.reindex(up.index, method="ffill")
            right_valid = valid.reindex(up.index, method="bfill")
            columns[column] = filled.where(left_valid & right_valid)
    upsampled = pd.DataFrame(columns)

    return upsampled


def data_gap_warnings(ts, source, variable):
    """EEWeatherWarnings for requested ranges the returned data does not
    cover: entirely empty series, late-starting or early-ending data, and
    long internal gaps."""
    warnings = []
    if len(ts) == 0:
        return warnings

    if ts.isna().all():
        warnings.append(
            EEWeatherWarning(
                qualified_name="eeweather.no_data_in_requested_range",
                description="No data was available within the requested range.",
                data={
                    "source": source,
                    "variable": variable,
                    "requested_start": ts.index[0].isoformat(),
                    "requested_end": ts.index[-1].isoformat(),
                },
            )
        )

        return warnings

    first_valid = ts.first_valid_index()
    leading_gap = first_valid - ts.index[0]
    if leading_gap > LEADING_GAP_WARNING_THRESHOLD:
        warnings.append(
            EEWeatherWarning(
                qualified_name="eeweather.data_starts_late",
                description=(
                    "Data begins {} after the start of the requested"
                    " range.".format(leading_gap)
                ),
                data={
                    "source": source,
                    "variable": variable,
                    "first_valid": first_valid.isoformat(),
                    "requested_start": ts.index[0].isoformat(),
                },
            )
        )

    last_valid = ts.last_valid_index()
    trailing_gap = ts.index[-1] - last_valid
    if trailing_gap > TRAILING_GAP_WARNING_THRESHOLD:
        warnings.append(
            EEWeatherWarning(
                qualified_name="eeweather.data_truncated",
                description=(
                    "Data ends {} before the end of the requested range.".format(
                        trailing_gap
                    )
                ),
                data={
                    "source": source,
                    "variable": variable,
                    "last_valid": last_valid.isoformat(),
                    "requested_end": ts.index[-1].isoformat(),
                },
            )
        )

    interior = ts.loc[first_valid:last_valid]
    if len(interior) > 1:
        period = interior.index[1] - interior.index[0]
        is_missing = interior.isna()
        max_gap_periods = int(is_missing.groupby((~is_missing).cumsum()).sum().max())
        max_gap = max_gap_periods * period
        if max_gap > INTERNAL_GAP_WARNING_THRESHOLD:
            warnings.append(
                EEWeatherWarning(
                    qualified_name="eeweather.data_gap",
                    description=(
                        "Data contains an internal gap of {}.".format(max_gap)
                    ),
                    data={
                        "source": source,
                        "variable": variable,
                        "max_gap_days": max_gap / timedelta(days=1),
                    },
                )
            )

    return warnings
