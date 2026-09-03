"""
Karachi AQI Forecasting Dashboard
==================================
Run with:  streamlit run dashboard/app.py

Production-style flow:
    Feast Online Store -> latest features -> loaded trained model -> 72h forecast -> dashboard

This dashboard does NOT read data/processed/*.csv or model_df directly. Current features are
retrieved through Feast's online store (feast_utils.get_online_features_for_prediction), and the
feature-engineering + AQI-categorization logic is imported from feature_engineering.py - the same
module the training notebook uses - so serving-time features are guaranteed to be defined
identically to training-time features (Feast's "define once, reuse for training and serving").

Prerequisite: run the notebook (notebooks/Karachi_AQI_Forecasting.ipynb) at least once through
the "Feast Feature Store Setup" -> "Materialize Features" sections, and through
"Save Model and Outputs", so that:
  - feature_store/ has an applied Feast registry + a materialized online store, and
  - models/ has a saved model + model_metadata.json / feature_columns.json
"""

import os
import sys
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# --------------------------------------------------------------------------
# Shared project modules (single source of truth - see module docstrings)
# --------------------------------------------------------------------------
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from feature_engineering import (  # noqa: E402
    create_features, aqi_to_category, aqi_health_message, AQI_CATEGORY_COLORS,
    WEATHER_HOURLY_VARS, AIR_QUALITY_HOURLY_VARS, WEATHER_COLS, POLLUTANT_COLS,
    EXTRA_WEATHER_PASSTHROUGH_COLS,
)
from data_pipeline import (  # noqa: E402
    LATITUDE, LONGITUDE, CITY_NAME,
    fetch_recent_weather_and_air_quality, merge_datasets,
)
import feast_utils as fu  # noqa: E402

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

FORECAST_HOURS = 72


# --------------------------------------------------------------------------
# Model + metadata loading
# --------------------------------------------------------------------------
@st.cache_resource
def load_production_model():
    """Loads the saved model + the feature schema it was trained with. The feature
    ORDER recorded here is what get_online_features_for_prediction() reindexes Feast's
    response to, before calling the model - this is what prevents a feature-order
    mismatch between training and serving."""
    config_path = os.path.join(MODELS_DIR, "model_config.json")
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    features_path = os.path.join(MODELS_DIR, "feature_columns.json")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    if not os.path.exists(config_path):
        return None

    config = json.load(open(config_path))
    metadata = json.load(open(metadata_path)) if os.path.exists(metadata_path) else {}
    feature_cols = json.load(open(features_path))
    scaler = joblib.load(scaler_path)

    if config["final_model_name"] == "Weighted Ensemble":
        weights = json.load(open(os.path.join(MODELS_DIR, "ensemble_weights.json")))
        models = {}
        for name in weights:
            safe = name.replace(" ", "_").replace("(", "").replace(")", "")
            keras_path = os.path.join(MODELS_DIR, f"{safe}.keras")
            pkl_path = os.path.join(MODELS_DIR, f"{safe}.pkl")
            if os.path.exists(pkl_path):
                models[name] = joblib.load(pkl_path)
            elif os.path.exists(keras_path):
                from tensorflow import keras as tfk
                models[name] = tfk.models.load_model(keras_path)
        return {"type": "ensemble", "models": models, "weights": weights, "scaler": scaler,
                "feature_cols": feature_cols, "config": config, "metadata": metadata}
    else:
        model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
        return {"type": "single", "model": model, "scaler": scaler,
                "feature_cols": feature_cols, "config": config, "metadata": metadata}


def predict_row(artifacts, X_row_df):
    if artifacts["type"] == "single":
        name = artifacts["config"]["final_model_name"]
        Xr = pd.DataFrame(artifacts["scaler"].transform(X_row_df), columns=X_row_df.columns) if "Ridge" in name else X_row_df
        return float(artifacts["model"].predict(Xr)[0])
    else:
        total = 0.0
        for name, w in artifacts["weights"].items():
            Xr = pd.DataFrame(artifacts["scaler"].transform(X_row_df), columns=X_row_df.columns) if "Ridge" in name else X_row_df
            total += w * float(artifacts["models"][name].predict(Xr)[0])
        return total


# --------------------------------------------------------------------------
# Live ingestion: fetch raw Open-Meteo data (data_pipeline.py - the same function the
# hourly CI/CD feature pipeline uses), engineer features + push into Feast (feast_utils.
# ingest_latest_observation() - also shared with the CI/CD feature pipeline), THEN read
# back through Feast's online store for the actual prediction. This keeps "the only
# path into the model" running through Feast, per the internship requirement - the
# freshly computed row is never used directly for prediction, only to update Feast.
# Neither the fetch nor the Feast-ingest logic is duplicated here: both come from the
# same shared modules scripts/feature_pipeline.py uses.
# --------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_recent_and_forecast():
    return fetch_recent_weather_and_air_quality(past_days=10, weather_forecast_days=4, air_forecast_days=1)


def update_feast_with_latest_observations(weather_df, air_df, feature_cols):
    """Thin wrapper around feast_utils.ingest_latest_observation() - the single shared
    implementation also used by scripts/feature_pipeline.py (the hourly GitHub Actions
    job). Kept as a dashboard-local name for readability at the call site below."""
    observed, future_weather, _latest_row = fu.ingest_latest_observation(
        weather_df, air_df,
        create_features_fn=create_features,
        merge_datasets_fn=merge_datasets,
        feature_cols=feature_cols,
    )
    return observed, future_weather


def run_recursive_forecast_from_feast(artifacts, seed_feature_vector, weather_history, future_weather):
    """72-hour recursive forecast, seeded from the feature vector retrieved from Feast's
    ONLINE store (not from a raw DataFrame read) for the first prediction, then walking
    forward exactly like the notebook's forecast_next_72_hours()."""
    feature_cols = artifacts["feature_cols"]
    last_ts = weather_history["timestamp"].max()
    working = weather_history.copy()
    last_known_pollutants = {c: working[c].iloc[-1] for c in POLLUTANT_COLS if c in working.columns}

    rows = []
    for step in range(1, FORECAST_HOURS + 1):
        next_ts = last_ts + timedelta(hours=step)
        new_row = {"timestamp": next_ts}
        if step - 1 < len(future_weather):
            wr = future_weather.iloc[step - 1]
            for c in WEATHER_COLS + EXTRA_WEATHER_PASSTHROUGH_COLS:
                if c in wr:
                    new_row[c] = wr[c]
        else:
            for c in WEATHER_COLS:
                if c in working.columns:
                    new_row[c] = working[c].iloc[-1]
        for c, v in last_known_pollutants.items():
            new_row[c] = v
        new_row["us_aqi"] = np.nan

        working = pd.concat([working, pd.DataFrame([new_row])], ignore_index=True)
        recomputed = create_features(working)
        last_row = recomputed.iloc[[-1]]

        if step == 1:
            # First step: use the feature vector actually retrieved from Feast's online
            # store (seed_feature_vector), not the freshly recomputed row - this is what
            # makes the *first* prediction genuinely "loaded from the Feature Store".
            X_next = seed_feature_vector
        else:
            X_next = last_row.reindex(columns=feature_cols, fill_value=0).fillna(method="ffill", axis=0).fillna(0)

        pred = float(np.clip(predict_row(artifacts, X_next), 0, 500))
        working.loc[working.index[-1], "us_aqi"] = pred

        forecast_day = (step - 1) // 24 + 1
        rows.append({
            "timestamp": next_ts, "predicted_aqi": round(pred, 1),
            "aqi_category": aqi_to_category(pred),
            "forecast_day": f"Day {forecast_day}", "forecast_hour": next_ts.strftime("%H:%M"),
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Karachi AQI Forecasting System", page_icon="🌫️", layout="wide")

st.title("🌫️ Karachi AQI Forecasting System")
st.caption("AI-based 72-hour hourly air quality forecast — features served from Feast, model loaded from disk")

artifacts = load_production_model()
if artifacts is None:
    st.error(
        "No trained model found in `models/`. Run the notebook "
        "`notebooks/Karachi_AQI_Forecasting.ipynb` first (through Section 24, 'Save Model and Outputs') "
        "so that `models/model_config.json` and the model files exist."
    )
    st.stop()

feature_cols = artifacts["feature_cols"]

# ---------------- Feature Store status section ----------------
st.subheader("Feature Store")

# Pre-flight diagnostic BEFORE attempting any retrieval: if Feast isn't set up
# correctly, this gives one clear, actionable message (with the exact fix commands)
# instead of a raw sqlite3.OperationalError / Feast internals traceback reaching the
# UI. This is a real diagnosis, not a blanket try/except - see
# feast_utils.verify_feast_setup() for what each check actually verifies.
setup_check = fu.verify_feast_setup()

if not setup_check["ok"]:
    # Self-heal the single most common recoverable state: apply() succeeded (table
    # exists) but materialize() has never actually run. Rather than making the user
    # look up their own data's date range and run commands by hand, try filling that
    # in automatically - this is narrowly scoped (see bootstrap_materialize_if_needed's
    # docstring) and never attempts to fix anything else (missing table, stuck
    # MATERIALIZING, registry mismatch), which still surface as an actionable message.
    with st.spinner("Feast online store has no data yet - materializing from your existing feature history..."):
        attempted, setup_check = fu.bootstrap_materialize_if_needed()
    if attempted and setup_check["ok"]:
        st.success("Feast online store was empty - automatically materialized from your existing offline "
                    "feature history. Continuing normally.")

if not setup_check["ok"]:
    st.error("**Feast Feature Store setup is incomplete** - the dashboard cannot retrieve online features yet.")
    st.write(setup_check["message"])
    with st.expander("Diagnostic details"):
        st.json(setup_check["checks"])
    st.code(
        "import feast_utils as fu\n"
        "fu.reset_feast_online_state()   # only needed if recovering from a previously broken state\n"
        "fu.apply_feast_definitions()\n"
        "fu.materialize_feast_features(start_date, end_date)   # or materialize_incremental()",
        language="python",
    )
    st.info("Run the commands above (e.g. in a notebook cell or a Python shell in this project's "
            "environment), then reload this page.")
    st.stop()

feast_status_col, feast_detail_col = st.columns([1, 2])

online_df = fu.get_latest_features_from_feast()
feast_connected = True

with feast_status_col:
    st.success("Feature Store Status: Connected")

with feast_detail_col:
    ts_val = online_df["event_timestamp"].iloc[0] if "event_timestamp" in online_df.columns else "n/a"
    st.write(f"**Latest Feature Timestamp:** {ts_val}")

# ---------------- Fetch live data & keep Feast current ----------------
with st.spinner("Fetching live weather & air quality data from Open-Meteo..."):
    try:
        weather_df, air_df = fetch_recent_and_forecast()
        fetch_error = None
    except Exception as e:
        fetch_error = str(e)
        weather_df = air_df = None

if fetch_error:
    st.error(f"Could not reach Open-Meteo: {fetch_error}")
    if not feast_connected:
        st.stop()
    st.info("Falling back to the last materialized Feast online features (no fresh observation pushed this run).")
    weather_history = None
    future_weather = None
else:
    with st.spinner("Updating Feast with the latest observation..."):
        try:
            weather_history, future_weather = update_feast_with_latest_observations(weather_df, air_df, feature_cols)
            # Re-pull online features now that materialize-incremental has run
            online_df = fu.get_latest_features_from_feast()
            feast_connected = True
        except Exception as e:
            st.warning(f"Could not update Feast with a fresh observation this run ({e}). "
                       f"Using whatever was last materialized.")
            weather_history = None
            future_weather = None

if not feast_connected or online_df is None:
    st.error(
        "Feast online store has no features available yet. In the notebook, run through "
        "'Materialize Features' at least once (Feast Feature Store Setup section) before "
        "using this dashboard."
    )
    st.stop()

# ---------------- Feature Store: display retrieved features ----------------
with st.expander("Features Retrieved from Feast", expanded=False):
    display_feats = ["us_aqi_lag_1", "us_aqi_rolling_mean_24", "temperature_2m",
                      "relative_humidity_2m", "surface_pressure", "wind_speed_10m"]
    display_feats = [c for c in display_feats if c in online_df.columns]
    table_rows = [{"Feature Name": c, "Value": round(float(online_df[c].iloc[0]), 2)} for c in display_feats]
    st.table(pd.DataFrame(table_rows))
    st.caption(f"{len(feature_cols)} total features available via the '{fu.FEATURE_SERVICE_NAME}' feature service "
               f"(showing a representative subset above).")

try:
    online_vector = fu.get_online_features_for_prediction(feature_cols)
except fu.FeastConfigError as e:
    st.error("**Feast online retrieval failed while preparing a prediction.**")
    st.write(str(e))
    st.info("Run `fu.verify_feast_setup()` for a full diagnostic, or the recovery commands "
            "shown in the Feature Store panel above.")
    st.stop()

# ---------------- Current conditions ----------------
current_aqi = float(online_df["us_aqi_lag_1"].iloc[0]) if "us_aqi_lag_1" in online_df.columns else None

st.subheader("Current Conditions")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Latest AQI", f"{current_aqi:.0f}" if current_aqi is not None else "n/a",
          aqi_to_category(current_aqi) if current_aqi is not None else "")
c2.metric("PM2.5", f"{online_df['pm2_5_lag_1'].iloc[0]:.1f} µg/m³" if "pm2_5_lag_1" in online_df.columns else "n/a")
c3.metric("PM10", f"{online_df['pm10_lag_1'].iloc[0]:.1f} µg/m³" if "pm10_lag_1" in online_df.columns else "n/a")
c4.metric("Temperature", f"{online_df['temperature_2m'].iloc[0]:.1f} °C" if "temperature_2m" in online_df.columns else "n/a")
c5.metric("Humidity", f"{online_df['relative_humidity_2m'].iloc[0]:.0f} %" if "relative_humidity_2m" in online_df.columns else "n/a")
c6.metric("Wind Speed", f"{online_df['wind_speed_10m'].iloc[0]:.1f} km/h" if "wind_speed_10m" in online_df.columns else "n/a")

if current_aqi is not None:
    st.info(f"**{aqi_to_category(current_aqi)}** — {aqi_health_message(current_aqi)}")

# ---------------- Forecast ----------------
if weather_history is not None and future_weather is not None and len(future_weather) > 0:
    with st.spinner("Generating 72-hour recursive forecast..."):
        try:
            forecast_df = run_recursive_forecast_from_feast(artifacts, online_vector, weather_history, future_weather)
            forecast_error = None
        except Exception as e:
            forecast_error = str(e)
            forecast_df = None
else:
    forecast_error = "No live weather forecast available this run (Open-Meteo fetch failed) - cannot run the 72h recursive forecast, only current conditions above."
    forecast_df = None

if forecast_error:
    st.warning(forecast_error)
elif forecast_df is not None:
    st.subheader("3-Day Forecast Summary")
    day_stats = forecast_df.groupby("forecast_day")["predicted_aqi"].agg(["mean", "min", "max"])
    cols = st.columns(3)
    for i, (day, row) in enumerate(day_stats.iterrows()):
        with cols[i]:
            st.metric(f"{day} Average AQI", f"{row['mean']:.0f}", aqi_to_category(row["mean"]))
            st.caption(f"Min {row['min']:.0f} · Max {row['max']:.0f}")

    overall_max = forecast_df["predicted_aqi"].max()
    overall_min = forecast_df["predicted_aqi"].min()
    st.warning(
        f"Peak forecasted AQI over the next 72 hours: **{overall_max:.0f}** "
        f"({aqi_to_category(overall_max)}). Lowest: **{overall_min:.0f}** ({aqi_to_category(overall_min)})."
    )

    st.subheader("72-Hour Hourly Forecast")
    fig = go.Figure()
    if weather_history is not None:
        fig.add_trace(go.Scatter(x=weather_history["timestamp"].tail(48), y=weather_history["us_aqi"].tail(48),
                                  mode="lines", name="Observed", line=dict(color="steelblue")))
    fig.add_trace(go.Scatter(x=forecast_df["timestamp"], y=forecast_df["predicted_aqi"],
                              mode="lines+markers", name="Forecast", line=dict(color="firebrick", dash="dash")))
    fig.update_layout(xaxis_title="Time", yaxis_title="US AQI", height=450, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View full 72-hour forecast table"):
        st.dataframe(forecast_df[["timestamp", "predicted_aqi", "aqi_category", "forecast_day", "forecast_hour"]],
                     use_container_width=True)

    if weather_history is not None:
        st.subheader("Recent Pollutant & Weather Trends")
        t1, t2 = st.columns(2)
        with t1:
            fig_pm = px.line(weather_history.tail(72), x="timestamp", y=["pm2_5", "pm10"],
                              title="PM2.5 & PM10 (last 72h)", labels={"value": "µg/m³", "timestamp": ""})
            st.plotly_chart(fig_pm, use_container_width=True)
        with t2:
            fig_w = px.line(weather_history.tail(72), x="timestamp", y=["temperature_2m", "relative_humidity_2m"],
                             title="Temperature & Humidity (last 72h)", labels={"value": "", "timestamp": ""})
            st.plotly_chart(fig_w, use_container_width=True)

# ---------------- Map ----------------
st.subheader("Location")
if HAS_FOLIUM:
    m = folium.Map(location=[LATITUDE, LONGITUDE], zoom_start=11)
    folium.Marker([LATITUDE, LONGITUDE], popup=f"{CITY_NAME} ({LATITUDE}, {LONGITUDE})",
                  tooltip="Forecast location").add_to(m)
    st_folium(m, width=700, height=350)
else:
    st.map(pd.DataFrame({"lat": [LATITUDE], "lon": [LONGITUDE]}))
    st.caption("Install `folium` and `streamlit-folium` for a richer map (`pip install folium streamlit-folium`).")

st.divider()
model_name = artifacts["config"]["final_model_name"]
aqi_standard = artifacts["config"]["aqi_standard"]
st.caption(
    f"Model: {model_name} (loaded from `models/`) · Features: Feast online store "
    f"(`{fu.FEATURE_SERVICE_NAME}`, {len(feature_cols)} features) · AQI standard: {aqi_standard} "
    f"· Location: {CITY_NAME} ({LATITUDE}, {LONGITUDE})"
)
