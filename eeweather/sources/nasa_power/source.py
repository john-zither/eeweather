"""The NASA POWER grid source served through the POWER hourly point API."""
import random
import time

from collections import namedtuple

import pandas as pd
import requests

from ...__version__ import __version__
from ..base import Source



API_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# the api rejects a submission carrying more parameters than this
MAX_PARAMETERS = 20

API_REQUEST_TRIES = 4

API_RETRY_BACKOFF_SECONDS = 2

API_TIMEOUT_SECONDS = 120

USER_AGENT = "eeweather/{} (+https://github.com/opendsm/eeweather)".format(__version__)

KPA_TO_HPA = 10.0

# hourly precipitation fields are served as mm/day rates, not hourly depths
PER_DAY_TO_HOURLY = 1.0 / 24.0

# native_unit is checked against every response so a silent upstream unit
# change fails loudly; scale multiplies a native value into the canonical unit
Parameter = namedtuple("Parameter", ["native", "canonical", "native_unit", "scale"])

# the meteorological grid (MERRA-2, spliced with GEOS-IT near real time)
MET_PARAMETERS = (
    Parameter("T2M", "temperature", "C", 1.0),
    Parameter("T2MDEW", "dew_point_temperature", "C", 1.0),
    Parameter("RH2M", "relative_humidity", "%", 1.0),
    Parameter("WS10M", "wind_speed", "m/s", 1.0),
    Parameter("QV2M", "specific_humidity", "g/kg", 1.0),
    Parameter("TS", "skin_temperature", "C", 1.0),
    Parameter("TSOIL1", "soil_temperature", "C", 1.0),
    Parameter("U10M", "eastward_wind", "m/s", 1.0),
    Parameter("V10M", "northward_wind", "m/s", 1.0),
    Parameter("Z0M", "surface_roughness", "m", 1.0),
    Parameter("PS", "surface_pressure", "kPa", KPA_TO_HPA),
    Parameter("PRECTOTCORR", "precipitation", "mm/day", PER_DAY_TO_HOURLY),
    Parameter("PRECSNO", "snowfall", "mm/day", PER_DAY_TO_HOURLY),
    Parameter("FRSNO", "snow_cover", "1", 1.0),
)

# the solar grid (CERES SYN1deg); irradiance arrives as Wh/m^2 over the hour,
# numerically identical to the hourly mean W/m2, so no scaling applies
SOLAR_PARAMETERS = (
    Parameter("ALLSKY_SFC_SW_DWN", "ghi", "Wh/m^2", 1.0),
    Parameter("CLRSKY_SFC_SW_DWN", "clearsky_ghi", "Wh/m^2", 1.0),
    Parameter("ALLSKY_SFC_SW_DNI", "dni", "Wh/m^2", 1.0),
    Parameter("CLRSKY_SFC_SW_DNI", "clearsky_dni", "Wh/m^2", 1.0),
    Parameter("ORIGINAL_ALLSKY_SFC_SW_DIFF", "dhi", "Wh/m^2", 1.0),
    Parameter("CLRSKY_SFC_SW_DIFF", "clearsky_dhi", "Wh/m^2", 1.0),
    Parameter("ORIGINAL_ALLSKY_SFC_SW_DIRH", "bhi", "Wh/m^2", 1.0),
    Parameter("CLRSKY_SFC_SW_DIRH", "clearsky_bhi", "Wh/m^2", 1.0),
    Parameter("ALLSKY_SRF_ALB", "albedo", "dimensionless", 1.0),
    Parameter("ALLSKY_SFC_LW_DWN", "longwave_down", "Wh/m^2", 1.0),
    Parameter("ALLSKY_SFC_LW_UP", "longwave_up", "Wh/m^2", 1.0),
    Parameter("AIRMASS", "airmass", "dimensionless", 1.0),
    Parameter("AOD_55", "aerosol_optical_depth_550", "dimensionless", 1.0),
    Parameter("AOD_84", "aerosol_optical_depth_840", "dimensionless", 1.0),
    Parameter("PW", "precipitable_water", "cm", 1.0),
    Parameter("CLOUD_AMT", "cloud_cover", "%", 1.0),
)

# each family is a distinct grid with its own geometry and publication latency
FAMILY_PARAMETERS = {"met": MET_PARAMETERS, "solar": SOLAR_PARAMETERS}

PARAMETERS = {
    parameter.native: parameter
    for parameter in MET_PARAMETERS + SOLAR_PARAMETERS
}

Block = namedtuple("Block", ["data", "sources", "api_version"])

# one session for connection reuse across the many per-point-year requests
_session = requests.Session()


def _get(url, params):  # pragma: no cover (mocked in tests via this seam)
    response = _session.get(
        url=url,
        params=params,
        timeout=API_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    )

    return response


def _retryable(status_code):
    """Whether another attempt could succeed: the documented rate limit and
    server faults, never a client error."""
    retryable = status_code == 429 or status_code >= 500

    return retryable


def _retry_delay(response, attempt):
    """Seconds to wait before the next attempt, honoring a Retry-After delay
    when the api sends one and otherwise backing off exponentially with
    jitter."""
    retry_after = response.headers.get("Retry-After", "")
    if retry_after.strip().isdigit():
        return float(retry_after)

    delay = API_RETRY_BACKOFF_SECONDS * 2**attempt
    jittered = delay * (1 + random.random())

    return jittered


def _error_messages(response):
    """The api's complaints about a request, empty when the body is not its
    json error document."""
    try:
        payload = response.json()
    except ValueError:
        return ()

    return tuple(payload.get("messages", ()))


def _rejection(status_code, messages):
    """A submission the api would not serve, quoting its own explanation."""
    error = ValueError(
        "The NASA POWER api served no data with status {}: {}".format(
            status_code, "; ".join(messages) or "no explanation given"
        )
    )

    return error


def _request(latitude, longitude, start, end, parameters):
    """Issue one POWER submission and return its parsed json body.

    Rate-limit and server responses are retried with growing backoff;
    client errors raise immediately, and a rejected submission raises a
    ValueError quoting the api's own messages rather than parsing an
    error document as data.
    """
    params = {
        "parameters": ",".join(parameters),
        "community": "RE",
        "latitude": latitude,
        "longitude": longitude,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
        # the api serves local solar time unless told otherwise
        "time-standard": "UTC",
    }

    for attempt in range(API_REQUEST_TRIES):
        response = _get(API_URL, params)
        if not _retryable(response.status_code) or attempt == API_REQUEST_TRIES - 1:
            break
        time.sleep(_retry_delay(response, attempt))

    if response.status_code >= 400:
        messages = _error_messages(response)
        if messages:
            raise _rejection(response.status_code, messages)
        response.raise_for_status()

    payload = response.json()
    # an error document carries messages in place of the data properties
    if "properties" not in payload:
        raise _rejection(response.status_code, payload.get("messages", ()))

    return payload


def _parse(payload, parameters):
    """Normalize a response into canonical columns on a UTC hourly index."""
    fill_value = payload["header"]["fill_value"]
    reported_units = payload["parameters"]
    values = payload["properties"]["parameter"]

    columns = {}
    for native in parameters:
        parameter = PARAMETERS[native]
        reported = reported_units[native]["units"]
        if reported != parameter.native_unit:
            raise ValueError(
                "The NASA POWER api served {} in '{}' where eeweather expects"
                " '{}'; the conversion to {} is no longer valid.".format(
                    native, reported, parameter.native_unit, parameter.canonical
                )
            )

        series = pd.Series(values[native], dtype=float)
        columns[parameter.canonical] = (
            series.mask(series == fill_value) * parameter.scale
        )

    df = pd.DataFrame(columns)
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H", utc=True)
    df = df.sort_index()

    return df


def _chunks(parameters):
    """Split a parameter list into submissions the api will accept."""
    for start in range(0, len(parameters), MAX_PARAMETERS):
        yield parameters[start:start + MAX_PARAMETERS]


class NASAPowerSource(Source):
    """Gridded weather from NASA POWER's hourly point API.

    Two independent grids serve the variables: a meteorological family
    (MERRA-2/GEOS-IT) and a solar family (CERES SYN1deg), each with its
    own cell geometry and publication latency. The api serves the value
    of the cell containing the requested point, without interpolation.
    """

    name = "nasa-power"
    kind = "observations"
    variables = tuple(
        parameter.canonical for parameter in MET_PARAMETERS + SOLAR_PARAMETERS
    )
    default_variables = ("temperature",)

    def fetch(self, latitude, longitude, start, end, parameters):
        """Fetch native POWER parameters for one grid point and date range.

        Parameters
        ----------
        latitude, longitude : float
            The point the api samples; POWER serves the containing cell.
        start, end : datetime.date
            Inclusive UTC date bounds of the request.
        parameters : sequence of str
            Native POWER parameter names, requested serially in chunks
            the api accepts and reassembled in the order given.

        Returns
        -------
        Block
            ``data`` is one column per parameter under its canonical
            name, indexed by UTC hour, with fill values as NaN and
            values converted to canonical units. ``sources`` and
            ``api_version`` come from the response header and identify
            the upstream products the values were built from.
        """
        parameters = tuple(parameters)
        frames = []
        sources = []
        api_version = None
        for chunk in _chunks(parameters):
            payload = _request(latitude, longitude, start, end, chunk)
            frames.append(_parse(payload, chunk))
            for source in payload["header"]["sources"]:
                if source not in sources:
                    sources.append(source)
            if api_version is None:
                api_version = payload["header"]["api"]["version"]

        data = pd.concat(frames, axis=1)
        block = Block(data, tuple(sources), api_version)

        return block
