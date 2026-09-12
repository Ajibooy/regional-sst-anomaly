# ==============================================================
# TROPICAL NORTH EAST ATLANTIC
# DAILY SST ANOMALY + REGIONAL CONFIRMED MHW OCCURRENCE
#
# Region:
#   0-30°N, 60-10°W
#
# Climatological baseline:
#   1981-2010
#
# Plot period:
#   1982-2024
#
# BACKGROUND:
#   Area-weighted regional daily SST anomaly
#
#   SST anomaly =
#   regional daily SST
#   -
#   regional daily climatological mean
#
# MHW DETECTION:
#   At EACH grid cell:
#
#   SST > daily P90
#   for >=5 consecutive days
#
# BLACK DOT:
#   Plotted when at least 10% of the ocean area/grid cells
#   are experiencing a CONFIRMED MHW on that day.
#
# threshold.nc:
#   C:\Users\Aina Ajibola\Desktop\P90_1981-2010\threshold.nc
#
# ==============================================================


# ==============================================================
# 0. IMPORT PACKAGES
# ==============================================================

import os
import glob
import gc
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")


# ==============================================================
# 1. USER SETTINGS
# ==============================================================

DATA_DIR = (
    r"C:\Users\Aina Ajibola\Desktop\oisst_data"
)

THRESHOLD_FILE = (
    r"C:\Users\Aina Ajibola\Desktop\P90_1981-2010\threshold.nc"
)


# --------------------------------------------------------------
# Study region
# --------------------------------------------------------------

LAT_MIN = 0.0
LAT_MAX = 30.0

LON_MIN = -60.0
LON_MAX = -10.0


# --------------------------------------------------------------
# Climatological baseline
# --------------------------------------------------------------

BASE_START = "1981-01-01"
BASE_END = "2010-12-31"


# --------------------------------------------------------------
# Plot period
# --------------------------------------------------------------

PLOT_START_YEAR = 1982
PLOT_END_YEAR = 2024


# --------------------------------------------------------------
# MHW parameters
# --------------------------------------------------------------

HALF_WINDOW = 5
SMOOTH_WINDOW = 31
MIN_MHW_DURATION = 5


# --------------------------------------------------------------
# Black-dot criterion
#
# 0.10 = 10% of regional ocean area
#
# You can later test:
# 0.05 = 5%
# 0.10 = 10%
# 0.20 = 20%
# --------------------------------------------------------------

MHW_AREA_FRACTION_THRESHOLD = 0.10


# --------------------------------------------------------------
# Memory control
# --------------------------------------------------------------

LAT_BLOCK_SIZE = 5


# --------------------------------------------------------------
# Heatmap colour scale
# --------------------------------------------------------------

COLOR_LIMIT = 2.0


# ==============================================================
# 2. CHECK INPUT PATHS
# ==============================================================

if not os.path.isdir(DATA_DIR):

    raise FileNotFoundError(
        f"OISST directory not found:\n{DATA_DIR}"
    )


if not os.path.isfile(THRESHOLD_FILE):

    raise FileNotFoundError(
        f"threshold.nc not found:\n{THRESHOLD_FILE}"
    )


print("=" * 90)
print("INPUT FILE CHECK")
print("=" * 90)

print(
    f"\nOISST directory:\n{DATA_DIR}"
)

print(
    f"\nThreshold file:\n{THRESHOLD_FILE}"
)


# ==============================================================
# 3. PREPROCESS OISST
# ==============================================================

def preprocess_oisst(ds):

    rename_dict = {}


    for old, new in {

        "latitude": "lat",
        "longitude": "lon",
        "Latitude": "lat",
        "Longitude": "lon",
        "Time": "time",
        "TIME": "time"

    }.items():

        if old in ds.coords or old in ds.dims:

            rename_dict[old] = new


    if rename_dict:

        ds = ds.rename(
            rename_dict
        )


    # ----------------------------------------------------------
    # Remove singleton vertical dimensions
    # ----------------------------------------------------------

    for dim in [

        "zlev",
        "depth",
        "lev",
        "level"

    ]:

        if (
            dim in ds.dims
            and
            ds.sizes[dim] == 1
        ):

            ds = ds.squeeze(
                dim,
                drop=True
            )


    if "sst" not in ds.data_vars:

        raise KeyError(
            "Variable 'sst' not found."
        )


    ds = ds[["sst"]]


    # ----------------------------------------------------------
    # Convert longitude:
    # 0-360 -> -180...180
    # ----------------------------------------------------------

    if float(ds.lon.max()) > 180:

        ds = ds.assign_coords(

            lon=(
                (ds.lon + 180.0) % 360.0
            ) - 180.0

        )


    ds = ds.sortby("lat")
    ds = ds.sortby("lon")


    # ----------------------------------------------------------
    # Select region
    # ----------------------------------------------------------

    ds = ds.sel(

        lat=slice(
            LAT_MIN,
            LAT_MAX
        ),

        lon=slice(
            LON_MIN,
            LON_MAX
        )

    )


    return ds


# ==============================================================
# 4. CLIMATOLOGICAL DAY
# ==============================================================

def get_clim_day(dates):

    dates = pd.DatetimeIndex(
        dates
    )


    reference_dates = pd.to_datetime(

        {

            "year":
                np.full(
                    len(dates),
                    2000
                ),

            "month":
                dates.month,

            "day":
                dates.day

        }

    )


    return (

        pd.DatetimeIndex(
            reference_dates
        )

        .dayofyear

        .to_numpy(
            dtype=np.int16
        )

    )


# ==============================================================
# 5. CIRCULAR 31-DAY SMOOTHING
# ==============================================================

def circular_smooth_1d(
    values,
    window=31
):

    values = np.asarray(
        values,
        dtype=np.float64
    )


    half = window // 2


    extended = np.concatenate(

        [
            values[-half:],
            values,
            values[:half]
        ]

    )


    smoothed = np.full(

        values.shape,

        np.nan,

        dtype=np.float64

    )


    for i in range(
        len(values)
    ):

        smoothed[i] = np.nanmean(

            extended[
                i:
                i + window
            ]

        )


    return smoothed


# ==============================================================
# 6. IDENTIFY CONFIRMED MHW DAYS
#
# Works on:
#
# time x cells
#
# Each grid cell is treated independently.
#
# ==============================================================

def identify_confirmed_mhw_days_2d(
    above_threshold,
    min_duration=5
):

    n_time, n_cells = (
        above_threshold.shape
    )


    confirmed = np.zeros(

        (
            n_time,
            n_cells
        ),

        dtype=bool

    )


    # ----------------------------------------------------------
    # Process every grid cell independently
    # ----------------------------------------------------------

    for cell in range(
        n_cells
    ):

        series = above_threshold[
            :,
            cell
        ]


        start = None


        for t in range(
            n_time
        ):

            is_above = bool(
                series[t]
            )


            # --------------------------------------------------
            # Start run
            # --------------------------------------------------

            if (
                is_above
                and
                start is None
            ):

                start = t


            # --------------------------------------------------
            # Finish run
            # --------------------------------------------------

            if start is not None:

                run_finished = (

                    (not is_above)

                    or

                    (
                        t
                        ==
                        n_time - 1
                    )

                )


                if run_finished:

                    if (
                        is_above
                        and
                        t == n_time - 1
                    ):

                        end = t

                    else:

                        end = t - 1


                    duration = (
                        end
                        -
                        start
                        +
                        1
                    )


                    if (
                        duration
                        >=
                        min_duration
                    ):

                        confirmed[
                            start:end + 1,
                            cell
                        ] = True


                    start = None


    return confirmed


# ==============================================================
# 7. FIND OISST FILES
# ==============================================================

files = sorted(

    glob.glob(

        os.path.join(
            DATA_DIR,
            "*_oisst.nc"
        )

    )

)


if not files:

    files = sorted(

        glob.glob(

            os.path.join(
                DATA_DIR,
                "*.nc"
            )

        )

    )


if not files:

    raise FileNotFoundError(
        f"No NetCDF files found in:\n{DATA_DIR}"
    )


print(
    "\n"
    + "=" * 90
)

print(
    "OISST FILES"
)

print(
    "=" * 90
)


print(
    f"\nFiles found: {len(files):,}"
)


print(
    f"First file: {os.path.basename(files[0])}"
)


print(
    f"Last file: {os.path.basename(files[-1])}"
)


# ==============================================================
# 8. OPEN OISST
# ==============================================================

print(
    "\nOpening OISST..."
)


ds = xr.open_mfdataset(

    files,

    combine="by_coords",

    preprocess=preprocess_oisst,

    parallel=False,

    data_vars="minimal",

    coords="minimal",

    compat="override",

    join="outer",

    engine="netcdf4"

)


ds = ds.sortby(
    "time"
)


sst = ds[
    "sst"
]


# ==============================================================
# 9. NORMALIZE TIME
# ==============================================================

time_index = pd.DatetimeIndex(
    sst.time.values
).normalize()


sst = sst.assign_coords(
    time=time_index
)


# --------------------------------------------------------------
# Remove duplicate dates
# --------------------------------------------------------------

keep = np.where(

    ~time_index.duplicated(
        keep="first"
    )

)[0]


sst = sst.isel(
    time=keep
)


sst = sst.sortby(
    "time"
)


all_dates = pd.DatetimeIndex(
    sst.time.values
)


print(
    f"\nActual OISST period: "
    f"{all_dates[0].date()} to "
    f"{all_dates[-1].date()}"
)


# ==============================================================
# 10. SST UNIT CHECK
# ==============================================================

sample = float(

    sst.isel(

        time=slice(
            0,
            min(
                10,
                sst.sizes["time"]
            )
        )

    )

    .mean(
        skipna=True
    )

    .compute()

)


if sample > 100:

    print(
        "\nConverting SST from Kelvin to °C..."
    )

    sst = (
        sst
        -
        273.15
    )


else:

    print(
        "\nSST already appears to be °C."
    )


# ==============================================================
# 11. AREA WEIGHTS
# ==============================================================

latitudes = sst.lat.values


weights_1d = np.cos(

    np.deg2rad(
        latitudes
    )

)


latitude_weights = xr.DataArray(

    weights_1d,

    coords={
        "lat":
            sst.lat
    },

    dims=[
        "lat"
    ]

)


# ==============================================================
# 12. REGIONAL DAILY SST
#
# Used for anomaly background only.
# ==============================================================

print(
    "\nCalculating regional area-weighted daily SST..."
)


regional_daily_sst = (

    sst

    .weighted(
        latitude_weights
    )

    .mean(

        dim=[
            "lat",
            "lon"
        ],

        skipna=True

    )

    .compute()

)


regional_daily_sst = regional_daily_sst.astype(
    np.float64
)


# ==============================================================
# 13. DAILY REGIONAL CLIMATOLOGICAL MEAN
#
# 1981-2010
# +/-5-day window
# 31-day smoothing
# ==============================================================

print(
    "\nCalculating 1981-2010 daily regional climatology..."
)


baseline_regional_sst = regional_daily_sst.sel(

    time=slice(
        BASE_START,
        BASE_END
    )

)


baseline_dates = pd.DatetimeIndex(

    baseline_regional_sst.time.values

)


baseline_values = np.asarray(

    baseline_regional_sst.values,

    dtype=np.float64

)


baseline_clim_days = get_clim_day(
    baseline_dates
)


daily_climatology = np.full(

    366,

    np.nan,

    dtype=np.float64

)


for day in range(
    1,
    367
):


    distance = np.abs(

        baseline_clim_days
        -
        day

    )


    distance = np.minimum(

        distance,

        366
        -
        distance

    )


    selected = (

        distance
        <=
        HALF_WINDOW

    )


    daily_climatology[
        day - 1
    ] = np.nanmean(

        baseline_values[
            selected
        ]

    )


daily_climatology = circular_smooth_1d(

    daily_climatology,

    window=SMOOTH_WINDOW

)


print(
    f"Climatology range: "
    f"{np.nanmin(daily_climatology):.2f} to "
    f"{np.nanmax(daily_climatology):.2f} °C"
)


# ==============================================================
# 14. CALCULATE DAILY REGIONAL SST ANOMALY
# ==============================================================

all_clim_days = get_clim_day(
    all_dates
)


regional_daily_values = np.asarray(

    regional_daily_sst.values,

    dtype=np.float64

)


climatology_for_dates = daily_climatology[

    all_clim_days - 1

]


regional_sst_anomaly = (

    regional_daily_values

    -

    climatology_for_dates

)


# ==============================================================
# 15. OPEN threshold.nc
# ==============================================================

print(
    "\nOpening threshold.nc..."
)


threshold_ds = xr.open_dataset(
    THRESHOLD_FILE
)


# --------------------------------------------------------------
# Standardize coordinate names
# --------------------------------------------------------------

rename_thr = {}


for old, new in {

    "latitude": "lat",
    "longitude": "lon",
    "Latitude": "lat",
    "Longitude": "lon"

}.items():

    if (
        old in threshold_ds.coords
        or
        old in threshold_ds.dims
    ):

        rename_thr[
            old
        ] = new


if rename_thr:

    threshold_ds = threshold_ds.rename(
        rename_thr
    )


# ==============================================================
# 16. FIND THRESHOLD VARIABLE
# ==============================================================

candidate_variables = []


for variable in threshold_ds.data_vars:

    lower = variable.lower()


    if any(

        key in lower

        for key in [
            "threshold",
            "thresh",
            "p90",
            "percentile"
        ]

    ):

        candidate_variables.append(
            variable
        )


if candidate_variables:

    THRESHOLD_VARIABLE = (
        candidate_variables[0]
    )


elif len(
    threshold_ds.data_vars
) == 1:

    THRESHOLD_VARIABLE = list(
        threshold_ds.data_vars
    )[0]


else:

    raise ValueError(

        "Could not automatically identify "
        "the P90 variable.\n"

        f"Variables:\n"
        f"{list(threshold_ds.data_vars)}"

    )


print(
    f"P90 variable: "
    f"{THRESHOLD_VARIABLE}"
)


threshold = threshold_ds[
    THRESHOLD_VARIABLE
].squeeze(
    drop=True
)


# ==============================================================
# 17. STANDARDIZE THRESHOLD LONGITUDE
# ==============================================================

if float(
    threshold.lon.max()
) > 180:

    threshold = threshold.assign_coords(

        lon=(
            (threshold.lon + 180.0) % 360.0
        ) - 180.0

    )


threshold = threshold.sortby(
    "lat"
)


threshold = threshold.sortby(
    "lon"
)


threshold = threshold.sel(

    lat=slice(
        LAT_MIN,
        LAT_MAX
    ),

    lon=slice(
        LON_MIN,
        LON_MAX
    )

)


# ==============================================================
# 18. FIND DAILY DIMENSION
# ==============================================================

day_dim = None


for dim in [

    "clim_day",
    "dayofyear",
    "day_of_year",
    "doy",
    "day",
    "time"

]:

    if (
        dim in threshold.dims
        and
        threshold.sizes[dim] in [
            365,
            366
        ]
    ):

        day_dim = dim

        break


if day_dim is None:

    for dim in threshold.dims:

        if threshold.sizes[dim] in [
            365,
            366
        ]:

            day_dim = dim

            break


if day_dim is None:

    raise ValueError(
        "Could not identify daily dimension in threshold.nc"
    )


if threshold.sizes[
    day_dim
] != 366:

    raise ValueError(

        "This script expects your corrected "
        "366-day threshold.nc."

    )


# ==============================================================
# 19. ALIGN THRESHOLD TO OISST GRID
# ==============================================================

same_lat = (

    threshold.sizes["lat"]
    ==
    sst.sizes["lat"]

    and

    np.allclose(
        threshold.lat.values,
        sst.lat.values
    )

)


same_lon = (

    threshold.sizes["lon"]
    ==
    sst.sizes["lon"]

    and

    np.allclose(
        threshold.lon.values,
        sst.lon.values
    )

)


if not (
    same_lat
    and
    same_lon
):

    print(
        "Aligning threshold grid to OISST grid..."
    )


    threshold = threshold.interp(

        lat=sst.lat,

        lon=sst.lon,

        method="nearest"

    )


threshold = threshold.transpose(

    day_dim,

    "lat",

    "lon"

)


# ==============================================================
# 20. CONVERT THRESHOLD TO STANDARD 366-DAY ARRAY
# ==============================================================

threshold_raw = np.asarray(

    threshold.values,

    dtype=np.float32

)


day_coordinate = np.asarray(

    threshold[
        day_dim
    ].values

)


n_lat = sst.sizes["lat"]
n_lon = sst.sizes["lon"]


threshold_366 = np.full(

    (
        366,
        n_lat,
        n_lon
    ),

    np.nan,

    dtype=np.float32

)


# --------------------------------------------------------------
# Datetime coordinate
# --------------------------------------------------------------

if np.issubdtype(

    day_coordinate.dtype,

    np.datetime64

):


    threshold_dates = pd.DatetimeIndex(
        day_coordinate
    )


    threshold_clim_days = get_clim_day(
        threshold_dates
    )


    for index, clim_day in enumerate(
        threshold_clim_days
    ):

        threshold_366[
            clim_day - 1
        ] = threshold_raw[
            index
        ]


# --------------------------------------------------------------
# Numeric 1-366 coordinate
# --------------------------------------------------------------

elif (
    np.issubdtype(
        day_coordinate.dtype,
        np.number
    )

    and

    np.nanmin(
        day_coordinate
    ) >= 1

    and

    np.nanmax(
        day_coordinate
    ) <= 366
):


    for index, clim_day in enumerate(

        day_coordinate.astype(
            int
        )

    ):

        threshold_366[
            clim_day - 1
        ] = threshold_raw[
            index
        ]


# --------------------------------------------------------------
# Otherwise assume already Jan-Dec order
# --------------------------------------------------------------

else:

    threshold_366[:] = threshold_raw


if np.all(
    np.isnan(
        threshold_366
    )
):

    raise ValueError(
        "P90 threshold contains only NaN values."
    )


# ==============================================================
# 21. OCEAN MASK
# ==============================================================

ocean_mask = np.isfinite(

    sst.sel(

        time="2000-01-01",

        method="nearest"

    ).values

)


# ==============================================================
# 22. SPATIAL AREA WEIGHTS
#
# Used to calculate daily confirmed-MHW area fraction.
# ==============================================================

weights_2d = np.broadcast_to(

    weights_1d[:, None],

    (
        n_lat,
        n_lon
    )

).astype(
    np.float64
)


weights_2d = np.where(

    ocean_mask,

    weights_2d,

    np.nan

)


total_ocean_weight = np.nansum(
    weights_2d
)


# ==============================================================
# 23. OUTPUT:
#
# Confirmed MHW area fraction for every date.
# ==============================================================

n_time = len(
    all_dates
)


mhw_area_fraction = np.zeros(

    n_time,

    dtype=np.float32

)


# ==============================================================
# 24. PROCESS EACH LATITUDE BLOCK
#
# Detect confirmed MHWs independently at each grid cell.
# ==============================================================

total_blocks = int(

    np.ceil(
        n_lat
        /
        LAT_BLOCK_SIZE
    )

)


print(
    f"\nDetecting grid-cell MHWs in "
    f"{total_blocks} latitude blocks..."
)


for block_number, block_start in enumerate(

    range(
        0,
        n_lat,
        LAT_BLOCK_SIZE
    ),

    start=1

):


    block_end = min(

        block_start
        +
        LAT_BLOCK_SIZE,

        n_lat

    )


    block_lat = (
        block_end
        -
        block_start
    )


    print(
        f"Block {block_number}/{total_blocks}: "
        f"rows {block_start + 1}-{block_end}"
    )


    # ----------------------------------------------------------
    # Load daily SST for block
    # ----------------------------------------------------------

    sst_block = np.asarray(

        sst.isel(

            lat=slice(
                block_start,
                block_end
            )

        ).values,

        dtype=np.float32

    )


    # ----------------------------------------------------------
    # Match daily P90 to each real date
    # ----------------------------------------------------------

    p90_block = threshold_366[

        all_clim_days - 1,

        block_start:block_end,

        :

    ]


    # ----------------------------------------------------------
    # Valid data
    # ----------------------------------------------------------

    valid = (

        np.isfinite(
            sst_block
        )

        &

        np.isfinite(
            p90_block
        )

    )


    # ----------------------------------------------------------
    # SST > P90
    # ----------------------------------------------------------

    above = (

        valid

        &

        (
            sst_block
            >
            p90_block
        )

    )


    # ----------------------------------------------------------
    # Flatten block:
    #
    # time x grid cells
    # ----------------------------------------------------------

    above_flat = above.reshape(

        n_time,

        block_lat
        *
        n_lon

    )


    # ----------------------------------------------------------
    # Confirm >=5 consecutive days
    # ----------------------------------------------------------

    confirmed_flat = identify_confirmed_mhw_days_2d(

        above_flat,

        min_duration=MIN_MHW_DURATION

    )


    # ----------------------------------------------------------
    # Restore:
    #
    # time x lat x lon
    # ----------------------------------------------------------

    confirmed_block = confirmed_flat.reshape(

        n_time,

        block_lat,

        n_lon

    )


    # ----------------------------------------------------------
    # Block weights
    # ----------------------------------------------------------

    block_weights = weights_2d[

        block_start:block_end,

        :

    ]


    # ----------------------------------------------------------
    # Area contribution from this block for each day
    # ----------------------------------------------------------

    weighted_confirmed = (

        confirmed_block

        *

        block_weights[
            None,
            :,
            :
        ]

    )


    # ----------------------------------------------------------
    # Add to total fraction numerator
    #
    # Temporarily store numerator in mhw_area_fraction
    # ----------------------------------------------------------

    mhw_area_fraction += np.nansum(

        weighted_confirmed,

        axis=(
            1,
            2
        )

    ).astype(
        np.float32
    )


    del sst_block
    del p90_block
    del valid
    del above
    del above_flat
    del confirmed_flat
    del confirmed_block
    del weighted_confirmed

    gc.collect()


# ==============================================================
# 25. CONVERT NUMERATOR TO AREA FRACTION
# ==============================================================

mhw_area_fraction = (

    mhw_area_fraction

    /

    total_ocean_weight

)


print(
    "\nMHW area fraction calculated."
)


print(
    f"Maximum confirmed-MHW area fraction: "
    f"{np.nanmax(mhw_area_fraction) * 100:.1f}%"
)


# ==============================================================
# 26. BLACK DOT CRITERION
#
# Black dot if >=10% regional ocean area is in confirmed MHW.
# ==============================================================

regional_mhw_indicator = (

    mhw_area_fraction

    >=

    MHW_AREA_FRACTION_THRESHOLD

)


print(
    f"\nDays with >= "
    f"{MHW_AREA_FRACTION_THRESHOLD * 100:.0f}% "
    f"of regional area in confirmed MHW:"
)


print(
    f"{np.sum(regional_mhw_indicator):,}"
)


# ==============================================================
# 27. SELECT 1982-2024
# ==============================================================

plot_mask = (

    (all_dates.year >= PLOT_START_YEAR)

    &

    (all_dates.year <= PLOT_END_YEAR)

)


plot_dates = all_dates[
    plot_mask
]


plot_anomaly = regional_sst_anomaly[
    plot_mask
]


plot_mhw_indicator = regional_mhw_indicator[
    plot_mask
]


plot_clim_days = all_clim_days[
    plot_mask
]


plot_area_fraction = mhw_area_fraction[
    plot_mask
]


# ==============================================================
# 28. CREATE YEAR x DAY ANOMALY MATRIX
# ==============================================================

years = np.arange(

    PLOT_START_YEAR,

    PLOT_END_YEAR + 1

)


n_years = len(
    years
)


anomaly_matrix = np.full(

    (
        366,
        n_years
    ),

    np.nan,

    dtype=np.float64

)


mhw_x = []
mhw_y = []


for i in range(
    len(plot_dates)
):


    date = plot_dates[
        i
    ]


    year_index = (

        date.year

        -

        PLOT_START_YEAR

    )


    clim_day = int(

        plot_clim_days[
            i
        ]

    )


    anomaly_matrix[

        clim_day - 1,

        year_index

    ] = plot_anomaly[
        i
    ]


    # ----------------------------------------------------------
    # Black dot only when >=10% area is in confirmed MHW
    # ----------------------------------------------------------

    if plot_mhw_indicator[
        i
    ]:


        mhw_x.append(
            date.year
        )


        mhw_y.append(
            clim_day
        )


mhw_x = np.asarray(
    mhw_x,
    dtype=float
)


mhw_y = np.asarray(
    mhw_y,
    dtype=float
)


# ==============================================================
# 29. MONTH POSITIONS
# ==============================================================

month_names = [

    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec"

]


month_middle_days = []


month_start_days = []


for month in range(
    1,
    13
):


    middle = pd.Timestamp(

        year=2000,

        month=month,

        day=15

    )


    start = pd.Timestamp(

        year=2000,

        month=month,

        day=1

    )


    month_middle_days.append(
        middle.dayofyear
    )


    month_start_days.append(
        start.dayofyear
    )


# ==============================================================
# 30. PCOLORMESH EDGES
# ==============================================================

year_edges = np.arange(

    PLOT_START_YEAR - 0.5,

    PLOT_END_YEAR + 1.5,

    1

)


day_edges = np.arange(

    0.5,

    367.5,

    1

)


# ==============================================================
# 31. CREATE FIGURE
# ==============================================================

fig, ax = plt.subplots(

    figsize=(
        18,
        10
    )

)


# ==============================================================
# 32. SST ANOMALY BACKGROUND
# ==============================================================

mesh = ax.pcolormesh(

    year_edges,

    day_edges,

    anomaly_matrix,

    cmap="RdBu_r",

    vmin=-COLOR_LIMIT,

    vmax=COLOR_LIMIT,

    shading="flat"

)


# ==============================================================
# 33. BLACK DOTS
#
# Confirmed MHW occurrence over >=10% regional area.
# ==============================================================

ax.scatter(

    mhw_x,

    mhw_y,

    s=3,

    c="black",

    marker=".",

    linewidths=0,

    zorder=5,

    label=(
        "Confirmed MHW area ≥ "
        f"{MHW_AREA_FRACTION_THRESHOLD * 100:.0f}%"
    )

)


# ==============================================================
# 34. X AXIS
# ==============================================================

ax.set_xlim(

    PLOT_START_YEAR - 0.5,

    PLOT_END_YEAR + 0.5

)


ax.set_xticks(
    years
)


ax.set_xticklabels(

    years,

    rotation=90,

    fontsize=8

)


ax.set_xlabel(

    "Year",

    fontsize=12,

    fontweight="bold"

)


# ==============================================================
# 35. Y AXIS
# ==============================================================

ax.set_ylim(
    0.5,
    366.5
)


ax.set_yticks(
    month_middle_days
)


ax.set_yticklabels(

    month_names,

    fontsize=10

)


ax.set_ylabel(

    "Month",

    fontsize=12,

    fontweight="bold"

)


# ==============================================================
# 36. MONTH BOUNDARIES
# ==============================================================

for month_start in month_start_days[
    1:
]:

    ax.axhline(

        month_start - 0.5,

        color="white",

        linewidth=0.25,

        alpha=0.35

    )


# ==============================================================
# 37. TITLE
# ==============================================================

ax.set_title(

    "Daily SST Anomaly and Confirmed Marine Heatwave Occurrence\n"
    "Tropical North East Atlantic (1982–2024)",

    fontsize=15,

    fontweight="bold",

    pad=15

)


# ==============================================================
# 38. COLORBAR
# ==============================================================

cbar = fig.colorbar(

    mesh,

    ax=ax,

    orientation="vertical",

    pad=0.025,

    fraction=0.035

)


cbar.set_label(

    "SST Anomaly (°C)",

    fontsize=12,

    fontweight="bold"

)


cbar.ax.yaxis.set_major_formatter(

    mticker.FormatStrFormatter(
        "%.1f"
    )

)


# ==============================================================
# 39. LEGEND
# ==============================================================

ax.legend(

    loc="upper left",

    fontsize=9,

    frameon=True,

    markerscale=3

)


# ==============================================================
# 40. GENERAL FORMATTING
# ==============================================================

ax.tick_params(

    direction="out",

    width=0.8

)


for spine in ax.spines.values():

    spine.set_linewidth(
        0.8
    )


plt.tight_layout()


# ==============================================================
# 41. SHOW
# ==============================================================

plt.show()


# ==============================================================
# 42. CLOSE
# ==============================================================

threshold_ds.close()

ds.close()


# ==============================================================
# 43. FINAL SUMMARY
# ==============================================================

print(
    "\n"
    + "=" * 90
)


print(
    "ANALYSIS COMPLETED"
)


print(
    "=" * 90
)


print(
    f"\nFigure period: "
    f"{PLOT_START_YEAR}-{PLOT_END_YEAR}"
)


print(
    "\nBackground:"
)


print(
    "Area-weighted regional SST anomaly "
    "relative to 1981-2010 daily climatology."
)


print(
    "\nGrid-cell MHW definition:"
)


print(
    "SST > daily P90 for >=5 consecutive days."
)


print(
    "\nBlack-dot definition:"
)


print(
    f"At least "
    f"{MHW_AREA_FRACTION_THRESHOLD * 100:.0f}% "
    f"of the regional ocean area is experiencing "
    f"a confirmed MHW."
)