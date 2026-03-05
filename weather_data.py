"""
Weather Data Enrichment using Copernicus ERA5 Reanalysis (0.25° grid).

Variables added per row:
  latitude, longitude, t (°C), tp (mm), ssr (MJ/m²), r (%)

Methodology (following reference paper):
  - ERA5 hourly reanalysis at 0.25° × 0.25° resolution
  - Each mandi snapped to nearest 0.25° grid point
  - If equidistant from multiple grid points, values are averaged
  - RH computed from 2m temperature & 2m dewpoint via Magnus formula
"""

import os
import json
import glob
import time
import logging
import calendar
import zipfile
import tempfile
import shutil

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_expanded")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "final_data")
GEOCODE_CACHE = os.path.join(os.path.dirname(__file__), "geocode_cache.json")
ERA5_CACHE_DIR = os.path.join(os.path.dirname(__file__), "era5_cache")

INDIA_AREA = [37.5, 67.0, 6.0, 98.0]   # N, W, S, E
GRID_RES = 0.25                          # ERA5 native resolution

ERA5_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "surface_net_solar_radiation",
]

ERA5_HOURS = [f"{h:02d}:00" for h in range(24)]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Geocoding ────────────────────────────────────────────────────────────────

def load_geocode_cache() -> dict:
    if os.path.exists(GEOCODE_CACHE):
        with open(GEOCODE_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_geocode_cache(cache: dict):
    with open(GEOCODE_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def geocode_locations(locations: list[tuple[str, str, str]]) -> dict:
    """Geocode (State, District, Market) → {key: {lat, lon}} with caching."""
    cache = load_geocode_cache()
    geolocator = Nominatim(user_agent="agmarknet_weather_enrichment", timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.5,
                          max_retries=3, error_wait_seconds=5.0)

    pending = [(s, d, m) for s, d, m in locations if f"{m}|{d}|{s}" not in cache]
    total = len(pending)
    log.info("Geocoding %d new locations (%d already cached)",
             total, len(locations) - total)

    for idx, (state, district, market) in enumerate(pending, 1):
        key = f"{market}|{district}|{state}"
        queries = [
            f"{market}, {district}, {state}, India",
            f"{district}, {state}, India",
            f"{state}, India",
        ]

        result = None
        for q in queries:
            try:
                result = geocode(q)
            except Exception as e:
                log.warning("Geocode error for '%s': %s", q, e)
                time.sleep(5)
                continue
            if result:
                break

        if result:
            cache[key] = {"lat": result.latitude, "lon": result.longitude}
            log.info("[%d/%d] Geocoded %-50s → (%.4f, %.4f)",
                     idx, total, key, result.latitude, result.longitude)
        else:
            cache[key] = None
            log.warning("[%d/%d] Could not geocode: %s", idx, total, key)

        if idx % 50 == 0:
            save_geocode_cache(cache)
            log.info("Cache saved (%d / %d done)", idx, total)

    save_geocode_cache(cache)
    return cache


# ── 0.25° grid snapping ─────────────────────────────────────────────────────

def snap_to_grid(lat: float, lon: float) -> list[tuple[float, float]]:
    """
    Snap a mandi to the nearest 0.25° ERA5 grid point(s).
    If equidistant from multiple corners, all are returned for averaging.
    """
    lat_lo = np.floor(lat / GRID_RES) * GRID_RES
    lon_lo = np.floor(lon / GRID_RES) * GRID_RES

    corners = [
        (lat_lo,            lon_lo),
        (lat_lo,            lon_lo + GRID_RES),
        (lat_lo + GRID_RES, lon_lo),
        (lat_lo + GRID_RES, lon_lo + GRID_RES),
    ]

    dists = [np.sqrt((lat - c[0])**2 + (lon - c[1])**2) for c in corners]
    min_dist = min(dists)
    tolerance = 0.10 * GRID_RES  # 0.025°

    return [corners[i] for i in range(4) if dists[i] <= min_dist + tolerance]


# ── Relative humidity ────────────────────────────────────────────────────────

def compute_relative_humidity(t2m_c: np.ndarray, d2m_c: np.ndarray) -> np.ndarray:
    """Magnus formula:  RH = 100 × e^(17.625·Td/(243.04+Td)) / e^(17.625·T/(243.04+T))"""
    a, b = 17.625, 243.04
    rh = 100.0 * np.exp(a * d2m_c / (b + d2m_c)) / np.exp(a * t2m_c / (b + t2m_c))
    return np.clip(rh, 0, 100)


# ── ERA5 download ────────────────────────────────────────────────────────────

def _unzip_and_merge_era5(zip_path: str, merged_path: str):
    """Extract a CDS ZIP archive, merge the NetCDF files inside, save as one file."""
    tmpdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmpdir)
        nc_files = sorted(glob.glob(os.path.join(tmpdir, "*.nc")))
        datasets = [xr.open_dataset(f, engine="netcdf4") for f in nc_files]
        merged = xr.merge(datasets)
        merged.to_netcdf(merged_path)
        for ds in datasets:
            ds.close()
        merged.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def download_era5_month(year: int, month: int) -> str:
    """Download one month of ERA5 data over India at 0.25° (cached)."""
    os.makedirs(ERA5_CACHE_DIR, exist_ok=True)
    nc_path = os.path.join(ERA5_CACHE_DIR, f"era5_{year}_{month:02d}_merged.nc")

    if os.path.exists(nc_path):
        log.info("ERA5 cache hit: %s", nc_path)
        return nc_path

    # Check if raw ZIP already downloaded from a previous run
    zip_path = os.path.join(ERA5_CACHE_DIR, f"era5_{year}_{month:02d}.nc")
    if os.path.exists(zip_path):
        log.info("Found raw ZIP %s, merging …", zip_path)
        _unzip_and_merge_era5(zip_path, nc_path)
        log.info("Merged ERA5 data → %s", nc_path)
        return nc_path

    _, last_day = calendar.monthrange(year, month)
    days = [f"{d:02d}" for d in range(1, last_day + 1)]

    log.info("Downloading ERA5 data for %d-%02d …", year, month)
    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ERA5_VARIABLES,
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": ERA5_HOURS,
            "area": INDIA_AREA,
            "grid": [GRID_RES, GRID_RES],
            "format": "netcdf",
        },
        zip_path,
    )
    _unzip_and_merge_era5(zip_path, nc_path)
    log.info("Merged ERA5 data → %s", nc_path)
    return nc_path


# ── Weather extraction ───────────────────────────────────────────────────────

def extract_daily_weather(nc_path: str, lat: float, lon: float, target_date: str) -> dict:
    """
    Extract daily weather at the nearest 0.25° grid point(s).
    If equidistant from multiple points, averages their values.
    Returns {t, tp, ssr, r} or NaNs on failure.
    """
    try:
        ds = xr.open_dataset(nc_path, engine="netcdf4")
        grid_points = snap_to_grid(lat, lon)
        # New CDS API uses 'valid_time'; fall back to 'time' for legacy files
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"
        day_data = ds.sel({time_dim: target_date})

        t2m_all, d2m_all, tp_all, ssr_all = [], [], [], []
        for glat, glon in grid_points:
            pt = day_data.sel(latitude=glat, longitude=glon, method="nearest")
            t2m_all.append(pt["t2m"].values)
            d2m_all.append(pt["d2m"].values)
            tp_all.append(pt["tp"].values)
            ssr_all.append(pt["ssr"].values)

        t2m_k = np.nanmean(t2m_all, axis=0)
        d2m_k = np.nanmean(d2m_all, axis=0)
        tp_m  = np.nanmean(tp_all, axis=0)
        ssr_j = np.nanmean(ssr_all, axis=0)

        t_c   = t2m_k - 273.15
        d2m_c = d2m_k - 273.15
        rh    = compute_relative_humidity(t_c, d2m_c)

        ds.close()
        return {
            "t":   round(float(np.nanmean(t_c)), 2),
            "tp":  round(float(np.nansum(tp_m) * 1000.0), 2),
            "ssr": round(float(np.nansum(ssr_j) / 1e6), 4),
            "r":   round(float(np.nanmean(rh)), 2),
        }

    except Exception as e:
        log.error("Weather extraction failed (%.2f, %.2f, %s): %s",
                  lat, lon, target_date, e)
        return {"t": np.nan, "tp": np.nan, "ssr": np.nan, "r": np.nan}


# ── CSV loading & enrichment ─────────────────────────────────────────────────

def load_all_csvs() -> dict[str, pd.DataFrame]:
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "agmarknet_*_expanded.csv")))
    datasets = {}
    for fpath in csv_files:
        commodity = os.path.basename(fpath).replace("agmarknet_", "").replace("_data_expanded.csv", "")
        log.info("Loading %s …", fpath)
        df = pd.read_csv(fpath, parse_dates=["Date"])
        datasets[commodity] = df
    return datasets


def enrich_commodity(commodity: str, df: pd.DataFrame, geocode_cache: dict):
    """Add latitude, longitude, t, tp, ssr, r columns and save to final_data/."""

    df["_geo_key"] = df.apply(
        lambda row: f"{str(row.get('Market', '')).strip()}|"
                     f"{str(row.get('District', '')).strip()}|"
                     f"{str(row.get('State', '')).strip()}",
        axis=1,
    )

    df["latitude"] = df["_geo_key"].map(
        lambda k: geocode_cache.get(k, {}).get("lat") if geocode_cache.get(k) else np.nan
    )
    df["longitude"] = df["_geo_key"].map(
        lambda k: geocode_cache.get(k, {}).get("lon") if geocode_cache.get(k) else np.nan
    )

    df["_ym"] = df["Date"].dt.to_period("M")
    year_months = df["_ym"].unique()

    nc_paths: dict[str, str] = {}
    for ym in sorted(year_months):
        key = f"{ym.year}-{ym.month:02d}"
        nc_paths[key] = download_era5_month(ym.year, ym.month)

    # ── Vectorised weather extraction ────────────────────────────────────
    # 1. Map each unique mandi lat/lon to its 0.25° grid point(s)
    has_geo = df["latitude"].notna() & df["longitude"].notna()
    unique_locs = df.loc[has_geo, ["latitude", "longitude"]].drop_duplicates()

    grid_map: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for _, loc_row in unique_locs.iterrows():
        key = (round(loc_row["latitude"], 6), round(loc_row["longitude"], 6))
        if key not in grid_map:
            grid_map[key] = snap_to_grid(key[0], key[1])

    log.info("[%s] %d unique locations → %d grid-point sets",
             commodity, len(unique_locs), len(grid_map))

    # 2. For each month, open the NetCDF ONCE and extract all grid points × all days
    weather_lookup: dict[tuple[float, float, str], dict] = {}

    for ym_key, nc_path in nc_paths.items():
        ds = xr.open_dataset(nc_path, engine="netcdf4")
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"

        for (lat, lon), grid_points in grid_map.items():
            for glat, glon in grid_points:
                pt = ds.sel(latitude=glat, longitude=glon, method="nearest")

                t2m_vals = pt["t2m"].values
                d2m_vals = pt["d2m"].values
                tp_vals  = pt["tp"].values
                ssr_vals = pt["ssr"].values
                times    = pt[time_dim].values

                days = pd.DatetimeIndex(times).normalize()
                unique_days = days.unique()

                for day in unique_days:
                    mask = days == day
                    date_str = day.strftime("%Y-%m-%d")
                    lk = (round(lat, 6), round(lon, 6), date_str)

                    if lk not in weather_lookup:
                        weather_lookup[lk] = {"t2m": [], "d2m": [], "tp": [], "ssr": []}
                    weather_lookup[lk]["t2m"].append(t2m_vals[mask])
                    weather_lookup[lk]["d2m"].append(d2m_vals[mask])
                    weather_lookup[lk]["tp"].append(tp_vals[mask])
                    weather_lookup[lk]["ssr"].append(ssr_vals[mask])

        ds.close()
        log.info("[%s] Extracted grid data from %s", commodity, os.path.basename(nc_path))

    # 3. Reduce to final weather values
    weather_final: dict[tuple[float, float, str], dict] = {}
    for lk, arrays in weather_lookup.items():
        t2m_k = np.nanmean(np.concatenate(arrays["t2m"]))
        d2m_k = np.nanmean(np.concatenate(arrays["d2m"]))
        tp_m  = np.nanmean(np.concatenate(arrays["tp"]))
        ssr_j = np.nanmean(np.concatenate(arrays["ssr"]))

        t_c   = t2m_k - 273.15
        d2m_c = d2m_k - 273.15
        rh    = float(compute_relative_humidity(np.array([t_c]), np.array([d2m_c]))[0])

        weather_final[lk] = {
            "t":   round(float(t_c), 2),
            "tp":  round(float(tp_m * 1000.0), 2),
            "ssr": round(float(ssr_j / 1e6), 4),
            "r":   round(rh, 2),
        }

    log.info("[%s] %d unique (location, date) weather values computed", commodity, len(weather_final))

    # 4. Map back to DataFrame via merge (fast)
    df["_lat_r"]  = df["latitude"].round(6)
    df["_lon_r"]  = df["longitude"].round(6)
    df["_date_s"] = df["Date"].dt.strftime("%Y-%m-%d")

    lookup_df = pd.DataFrame([
        {"_lat_r": k[0], "_lon_r": k[1], "_date_s": k[2], **v}
        for k, v in weather_final.items()
    ])
    df = df.merge(lookup_df, on=["_lat_r", "_lon_r", "_date_s"], how="left")

    df.drop(columns=["_geo_key", "_ym", "_lat_r", "_lon_r", "_date_s"], inplace=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"agmarknet_{commodity}_data_final.csv")
    df.to_csv(out_path, index=False)
    log.info("Saved enriched data → %s  (%d rows)", out_path, len(df))
    return df


def main():
    log.info("=== Weather Data Enrichment Pipeline ===")

    datasets = load_all_csvs()
    if not datasets:
        log.error("No CSV files found in %s", DATA_DIR)
        return

    all_locations = set()
    for commodity, df in datasets.items():
        for _, row in df.iterrows():
            state = str(row.get("State", "")).strip()
            district = str(row.get("District", "")).strip()
            market = str(row.get("Market", "")).strip()
            if state and district and market:
                all_locations.add((state, district, market))

    log.info("Found %d unique (State, District, Market) locations", len(all_locations))

    geocode_cache = geocode_locations(list(all_locations))
    resolved = sum(1 for v in geocode_cache.values() if v is not None)
    log.info("Geocoded %d / %d locations", resolved, len(geocode_cache))

    for commodity, df in datasets.items():
        log.info("─── Enriching: %s (%d rows) ───", commodity, len(df))
        enrich_commodity(commodity, df, geocode_cache)

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
