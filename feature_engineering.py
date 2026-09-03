import numpy as np
import pandas as pd


WEATHER_HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "precipitation", "rain", "cloud_cover", "surface_pressure", "pressure_msl",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
]

AIR_QUALITY_HOURLY_VARS = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
    "ozone", "ammonia", "us_aqi", "us_aqi_pm2_5", "us_aqi_pm10", "us_aqi_no2",
    "us_aqi_co", "us_aqi_so2", "us_aqi_ozone",
]

WEATHER_COLS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
                "precipitation", "cloud_cover", "surface_pressure", "wind_speed_10m",
                "wind_direction_10m", "wind_gusts_10m"]

POLLUTANT_COLS = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]

EXTRA_WEATHER_PASSTHROUGH_COLS = ["rain", "pressure_msl",
                                   "shortwave_radiation", "direct_radiation", "diffuse_radiation"]


LAG_TARGET_HOURS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
LAG_POLLUTANT_HOURS = [1, 3, 6, 12, 24, 48, 72]
ROLLING_WINDOWS = [3, 6, 12, 24, 48]

TARGET_HORIZONS = (1, 24, 48, 72)

AQI_STANDARD = "US EPA AQI (0-500 scale)"

AQI_CATEGORY_COLORS = {
    "Good": "#00E400", "Moderate": "#FFFF00", "Unhealthy for Sensitive Groups": "#FF7E00",
    "Unhealthy": "#FF0000", "Very Unhealthy": "#8F3F97", "Hazardous": "#7E0023",
}



def add_lag_features(df, col, lags):
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df, col, windows):
    
    shifted = df[col].shift(1)
    for w in windows:
        df[f"{col}_rolling_mean_{w}"] = shifted.rolling(window=w, min_periods=max(2, w // 2)).mean()
        if w >= 12:
            df[f"{col}_rolling_std_{w}"] = shifted.rolling(window=w, min_periods=max(2, w // 2)).std()
            df[f"{col}_rolling_min_{w}"] = shifted.rolling(window=w, min_periods=max(2, w // 2)).min()
            df[f"{col}_rolling_max_{w}"] = shifted.rolling(window=w, min_periods=max(2, w // 2)).max()
    return df


def add_time_features(df, ts_col="timestamp"):
    ts = df[ts_col]
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_month"] = ts.dt.day
    df["month"] = ts.dt.month
    df["quarter"] = ts.dt.quarter
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["day_of_year"] = ts.dt.dayofyear
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_weather_engineered_features(df):
    for col in ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m"]:
        if col in df:
            df[f"{col}_change_1h"] = df[col].diff(1)
            df[f"{col}_change_3h"] = df[col].diff(3)

    if "temperature_2m" in df and "relative_humidity_2m" in df:
        df["temp_humidity_interaction"] = df["temperature_2m"] * df["relative_humidity_2m"]

    if "wind_direction_10m" in df:
        rad = np.deg2rad(df["wind_direction_10m"])
        df["wind_dir_sin"] = np.sin(rad)
        df["wind_dir_cos"] = np.cos(rad)

    for col in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]:
        if col in df:
            shifted = df[col].shift(1)
            df[f"{col}_rolling_mean_24"] = shifted.rolling(24, min_periods=6).mean()

    # Dispersion proxy: still, humid, high-pressure conditions tend to trap pollutants
    if all(c in df for c in ["wind_speed_10m", "relative_humidity_2m", "surface_pressure"]):
        df["low_dispersion_index"] = (
            (1 / (df["wind_speed_10m"].clip(lower=0.5))) * 0.5
            + (df["relative_humidity_2m"] / 100) * 0.3
            + (df["surface_pressure"] / df["surface_pressure"].mean()) * 0.2
        )
    return df


def create_features(df):
    """Full feature engineering pipeline. Input: cleaned hourly df with a 'timestamp' column
    and raw weather/pollutant columns (+ 'us_aqi' target column). Output: feature-rich df,
    still one row per hour, ready for target construction (training) or serving (dashboard)."""
    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    df = add_lag_features(df, "us_aqi", LAG_TARGET_HOURS)
    df = add_rolling_features(df, "us_aqi", ROLLING_WINDOWS)

    for col in POLLUTANT_COLS:
        if col in df.columns:
            df = add_lag_features(df, col, LAG_POLLUTANT_HOURS)
            df = add_rolling_features(df, col, [3, 6, 24])

    df = add_time_features(df)
    df = add_weather_engineered_features(df)

    return df


def create_targets(df, horizons=TARGET_HORIZONS):
    df = df.copy()
    for h in horizons:
        df[f"AQI_t+{h}"] = df["us_aqi"].shift(-h)
    return df


def get_feature_columns(model_df, target_horizons=TARGET_HORIZONS):
    """Given a fully-featured + targeted DataFrame, return the ordered list of model
    input feature columns (everything except identifiers/targets/non-numeric)."""
    exclude_cols = {"timestamp", "event_timestamp", "location_id", "created_timestamp", "us_aqi"} \
        | {f"AQI_t+{h}" for h in target_horizons}
    return [c for c in model_df.columns if c not in exclude_cols and model_df[c].dtype != "object"]


def aqi_to_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"


def aqi_health_message(aqi):
    cat = aqi_to_category(aqi)
    messages = {
        "Good": "Air quality is satisfactory and poses little or no risk.",
        "Moderate": "Air quality is acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion.",
        "Unhealthy for Sensitive Groups": "Members of sensitive groups (children, elderly, those with respiratory/heart conditions) may experience health effects. General public is less likely to be affected.",
        "Unhealthy": "Everyone may begin to experience health effects. Sensitive groups should limit prolonged outdoor exposure.",
        "Very Unhealthy": "Health alert: everyone may experience more serious health effects. Avoid prolonged outdoor exertion.",
        "Hazardous": "Health warning of emergency conditions. The entire population is likely to be affected - avoid all outdoor exertion.",
    }
    return messages[cat]
