

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from feature_engineering import WEATHER_HOURLY_VARS, AIR_QUALITY_HOURLY_VARS


LATITUDE = 24.8607
LONGITUDE = 67.0011
TIMEZONE = "Asia/Karachi"
CITY_NAME = "Karachi"

WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_ARCHIVE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_FORECAST_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"  # supports forecast too

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")



def _date_chunks(start_date, end_date, chunk_days=90):
    """Yield (chunk_start, chunk_end) date strings covering [start_date, end_date] inclusive."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = chunk_end + timedelta(days=1)


def _request_with_retries(url, params, max_retries=4, timeout=60):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 5 * attempt
                print(f"    Rate limited (429). Waiting {wait}s...")
                time.sleep(wait)
                last_err = f"HTTP 429: {resp.text[:200]}"
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                time.sleep(2 * attempt)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = str(e)
            print(f"    Connection issue on attempt {attempt}/{max_retries}: {e}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"Failed to fetch data from {url} after {max_retries} attempts. Last error: {last_err}")


def _fetch_hourly_chunked(base_url, hourly_vars, start_date, end_date, extra_params=None,
                           chunk_days=90, label="data", latitude=LATITUDE, longitude=LONGITUDE,
                           timezone=TIMEZONE):
    """Fetch hourly data in chunks and concatenate into a single DataFrame."""
    all_frames = []
    chunks = list(_date_chunks(start_date, end_date, chunk_days=chunk_days))
    print(f"Fetching {label}: {len(chunks)} chunk(s) from {start_date} to {end_date}")
    for i, (c_start, c_end) in enumerate(chunks, 1):
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": c_start,
            "end_date": c_end,
            "hourly": ",".join(hourly_vars),
            "timezone": timezone,
        }
        if extra_params:
            params.update(extra_params)
        print(f"  [{i}/{len(chunks)}] {c_start} -> {c_end}")
        data = _request_with_retries(base_url, params)
        if "hourly" not in data or "time" not in data.get("hourly", {}):
            raise RuntimeError(f"Malformed response for {label} chunk {c_start}->{c_end}: {data}")
        df_chunk = pd.DataFrame(data["hourly"])
        if df_chunk.empty:
            print(f"    WARNING: empty chunk for {c_start}->{c_end}, skipping.")
            continue
        all_frames.append(df_chunk)
        time.sleep(0.3)  # be polite to the free API

    if not all_frames:
        raise RuntimeError(f"No {label} could be downloaded for {start_date}..{end_date}. "
                            f"Check your internet connection and the Open-Meteo API status.")

    df = pd.concat(all_frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return df


def fetch_weather_data(start_date, end_date, force_download=False, data_raw_dir=DEFAULT_DATA_RAW_DIR):
    """Download (or load cached) historical hourly weather data for Karachi."""
    os.makedirs(data_raw_dir, exist_ok=True)
    cache_path = os.path.join(data_raw_dir, f"weather_{start_date}_{end_date}.csv")
    if os.path.exists(cache_path) and not force_download:
        print(f"Loading cached weather data: {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["time"])
        return df

    df = _fetch_hourly_chunked(WEATHER_ARCHIVE_URL, WEATHER_HOURLY_VARS, start_date, end_date, label="weather data")
    df.to_csv(cache_path, index=False)
    print(f"Saved weather data -> {cache_path}  ({len(df)} rows)")
    return df


def fetch_air_quality_data(start_date, end_date, force_download=False, data_raw_dir=DEFAULT_DATA_RAW_DIR):
    """Download (or load cached) historical hourly air-quality data for Karachi."""
    os.makedirs(data_raw_dir, exist_ok=True)
    cache_path = os.path.join(data_raw_dir, f"air_quality_{start_date}_{end_date}.csv")
    if os.path.exists(cache_path) and not force_download:
        print(f"Loading cached air quality data: {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["time"])
        return df

    
    df = _fetch_hourly_chunked(AIR_QUALITY_ARCHIVE_URL, AIR_QUALITY_HOURLY_VARS, start_date, end_date,
                                chunk_days=60, label="air quality data")
    df.to_csv(cache_path, index=False)
    print(f"Saved air quality data -> {cache_path}  ({len(df)} rows)")
    return df


def fetch_weather_forecast(forecast_days=4):
    """Fetch real forward-looking weather FORECAST (not historical) for the next N days.
    Legitimately available at prediction time - used as an input feature for future
    timesteps in the recursive 72-hour forecast (both the notebook's
    forecast_next_72_hours() and the dashboard's live forecast)."""
    params = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "forecast_days": forecast_days,
        "timezone": TIMEZONE,
    }
    data = _request_with_retries(WEATHER_FORECAST_URL, params)
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_air_quality_recent(past_days=8):
    """Fetch the most recent observed air-quality data (used to build lag/rolling
    features right up to 'now' - for live forecasting, the dashboard, and the hourly
    feature pipeline)."""
    params = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "hourly": ",".join(AIR_QUALITY_HOURLY_VARS),
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": TIMEZONE,
    }
    data = _request_with_retries(AIR_QUALITY_FORECAST_URL, params)
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_recent_weather_and_air_quality(past_days=10, weather_forecast_days=4, air_forecast_days=1):
    """Combined recent-observation + forward-forecast fetch used by both the dashboard
    and the hourly feature pipeline: past `past_days` of observed weather + air quality,
    plus a forward weather forecast (needed for the dashboard's recursive forecast) and
    a short air-quality forecast tail from Open-Meteo. Returns (weather_df, air_df) with
    a shared 'time' column, not yet merged/cleaned."""
    w_params = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "past_days": past_days, "forecast_days": weather_forecast_days, "timezone": TIMEZONE,
    }
    w_data = _request_with_retries(WEATHER_FORECAST_URL, w_params)
    weather_df = pd.DataFrame(w_data["hourly"])
    weather_df["time"] = pd.to_datetime(weather_df["time"])

    aq_params = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "hourly": ",".join(AIR_QUALITY_HOURLY_VARS),
        "past_days": past_days, "forecast_days": air_forecast_days, "timezone": TIMEZONE,
    }
    aq_data = _request_with_retries(AIR_QUALITY_FORECAST_URL, aq_params)
    air_df = pd.DataFrame(aq_data["hourly"])
    air_df["time"] = pd.to_datetime(air_df["time"])

    return weather_df, air_df



def merge_datasets(weather, air_quality):
    """Inner-join weather and air-quality data on the hourly timestamp."""
    merged = pd.merge(weather, air_quality, on="time", how="inner", suffixes=("", "_aq"))
    merged = merged.sort_values("time").reset_index(drop=True)
    merged = merged.rename(columns={"time": "timestamp"})
    return merged



def data_quality_report(df):
    report = {}

    report["duplicate_timestamps"] = int(df["timestamp"].duplicated().sum())

    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report["missing_by_column"] = missing_pct[missing_pct > 0].to_dict()

    full_range = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    missing_timestamps = full_range.difference(df["timestamp"])
    report["missing_hours_count"] = len(missing_timestamps)
    report["expected_hours"] = len(full_range)
    report["actual_hours"] = len(df)

    impossible = {}
    if "relative_humidity_2m" in df: impossible["humidity_out_of_range"] = int(((df["relative_humidity_2m"] < 0) | (df["relative_humidity_2m"] > 100)).sum())
    if "pm2_5" in df: impossible["negative_pm25"] = int((df["pm2_5"] < 0).sum())
    if "pm10" in df: impossible["negative_pm10"] = int((df["pm10"] < 0).sum())
    if "us_aqi" in df: impossible["aqi_out_of_range"] = int(((df["us_aqi"] < 0) | (df["us_aqi"] > 500)).sum())
    report["impossible_values"] = impossible

    return report



def clean_data(df, max_gap_hours=6):
    df = df.copy()
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")

    full_index = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    df = df.set_index("timestamp").reindex(full_index)
    df.index.name = "timestamp"

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df[numeric_cols] = df[numeric_cols].interpolate(method="time", limit=max_gap_hours, limit_direction="both")

    if "relative_humidity_2m" in df: df["relative_humidity_2m"] = df["relative_humidity_2m"].clip(0, 100)
    for col in ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "ammonia"]:
        if col in df: df[col] = df[col].clip(lower=0)
    if "us_aqi" in df: df["us_aqi"] = df["us_aqi"].clip(0, 500)

    for col in ["pm2_5", "pm10", "us_aqi"]:
        if col in df:
            lo, hi = df[col].quantile(0.001), df[col].quantile(0.999)
            df[col] = df[col].clip(lo, hi)

    df = df.reset_index()

    before = len(df)
    df = df.dropna(subset=["us_aqi"]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with un-recoverable missing AQI (long gaps beyond {max_gap_hours}h)")

    return df
