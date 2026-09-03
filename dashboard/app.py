import os
import sys
import json
import html
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



st.set_page_config(page_title="Karachi Air Intelligence — 72-Hour AQI Forecast", page_icon="🌫️",
                   layout="wide", initial_sidebar_state="expanded")


st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#0a0f1c;--panel:#101a30;--panel-2:#0c1424;--border:#1e2a44;--text:#e8eef8;--muted:#8fa3c6;--muted-2:#64748b;--accent:#2dd4bf;--accent-2:#22d3ee;--ok:#4ade80;--warn:#fbbf24;--danger:#f87171;}
html,body,[data-testid="stAppViewContainer"]{font-family:'Inter',-apple-system,'Segoe UI',sans-serif;background:radial-gradient(1200px 520px at 16% -12%,#0f1c36 0%,var(--bg) 55%) no-repeat fixed;}
[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer,[data-testid="stToolbar"]{visibility:hidden;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1526,var(--bg));border-right:1px solid var(--border);}
[data-testid="stSidebar"] .block-container{padding-top:1.5rem;}
[data-testid="stSidebar"] hr{border-color:var(--border);}
h1,h2,h3,h4,h5{color:var(--text);letter-spacing:-.02em;font-family:'Inter',sans-serif;}
.stMarkdown p{color:var(--muted);}
[data-testid="stExpander"]{border:1px solid var(--border);border-radius:12px;background:rgba(16,26,48,.55);overflow:hidden;}
[data-testid="stExpander"] summary{color:var(--text);font-weight:600;}
[data-testid="stAlert"]{border-radius:12px;border:1px solid var(--border);background:rgba(16,26,48,.6);color:var(--text);}
.stButton>button{background:linear-gradient(135deg,var(--accent),var(--accent-2));color:#04121f;font-weight:700;border:none;border-radius:10px;padding:.5rem 1.1rem;}
.stButton>button:hover{border:none;color:#04121f;}
[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:12px;overflow:hidden;}
[data-testid="stVerticalBlockBorderWrapper"]{background:linear-gradient(180deg,rgba(16,26,48,.7),rgba(12,20,36,.7));border:1px solid var(--border)!important;border-radius:16px;}

/* ---------- Header / branding ---------- */
.app-header{display:flex;justify-content:space-between;align-items:center;gap:1.2rem;margin:.2rem 0 1rem;flex-wrap:wrap;}
.brand{display:flex;align-items:center;gap:.85rem;}
.brand-mark{width:46px;height:46px;border-radius:13px;background:linear-gradient(135deg,rgba(45,212,191,.22),rgba(34,211,238,.10));border:1px solid rgba(45,212,191,.4);display:flex;align-items:center;justify-content:center;color:var(--accent);flex-shrink:0;}
.brand-mark svg{width:26px;height:26px;stroke:currentColor;fill:none;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round;}
.brand-title{font-size:1.5rem;font-weight:800;letter-spacing:-.02em;color:var(--text);line-height:1.15;}
.brand-sub{color:var(--muted);font-size:.86rem;font-weight:500;margin-top:.15rem;}
.header-meta{display:flex;flex-direction:column;align-items:flex-end;gap:.45rem;}
.live-pill{display:inline-flex;align-items:center;gap:.5rem;background:rgba(74,222,128,.12);color:var(--ok);border:1px solid rgba(74,222,128,.35);border-radius:999px;padding:.35rem .85rem;font-size:.74rem;font-weight:700;letter-spacing:.08em;}
.live-pill i{width:8px;height:8px;border-radius:50%;background:var(--ok);display:inline-block;animation:pulse 2s infinite;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(74,222,128,.55);}70%{box-shadow:0 0 0 8px rgba(74,222,128,0);}100%{box-shadow:0 0 0 0 rgba(74,222,128,0);}}
.meta-row{display:flex;gap:.45rem;flex-wrap:wrap;justify-content:flex-end;}
.meta-pill{display:inline-flex;align-items:center;gap:.4rem;background:rgba(16,26,48,.8);border:1px solid var(--border);border-radius:999px;padding:.3rem .75rem;font-size:.73rem;font-weight:600;color:var(--muted);letter-spacing:.02em;}
.meta-pill svg{width:13px;height:13px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;}

/* ---------- Section headers ---------- */
.sec-head{display:flex;align-items:center;gap:.8rem;margin:2rem 0 .9rem;}
.sec-bar{width:4px;height:26px;border-radius:3px;background:linear-gradient(180deg,var(--accent),var(--accent-2));flex-shrink:0;}
.sec-kicker{font-size:.66rem;font-weight:700;letter-spacing:.15em;color:var(--accent);text-transform:uppercase;}
.sec-title{font-size:1.22rem;font-weight:800;color:var(--text);letter-spacing:-.015em;margin-top:.04rem;}
.sec-sub{font-size:.82rem;color:var(--muted);margin-top:.1rem;}

/* ---------- Pipeline architecture strip ---------- */
.pipe-wrap{margin:.35rem 0 1rem;padding:.65rem .9rem;background:linear-gradient(180deg,rgba(16,26,48,.65),rgba(12,20,36,.65));border:1px solid var(--border);border-radius:14px;overflow-x:auto;}
.pipeline{display:flex;align-items:center;gap:.45rem;min-width:max-content;}
.pipe-node{font-size:.66rem;font-weight:700;letter-spacing:.1em;color:var(--muted);background:rgba(20,32,58,.9);border:1px solid var(--border);border-radius:8px;padding:.35rem .65rem;white-space:nowrap;}
.pipe-node.hot{color:#031a16;background:linear-gradient(135deg,var(--accent),var(--accent-2));border-color:transparent;}
.pipe-arrow{color:var(--accent-2);font-size:.8rem;font-weight:700;}

/* ---------- Hero AQI ---------- */
.hero-card{display:flex;align-items:stretch;justify-content:space-between;gap:1.4rem;background:linear-gradient(135deg,rgba(18,28,52,.96),rgba(11,18,34,.94));border:1px solid var(--border);border-left:5px solid var(--heroacc,#2dd4bf);border-radius:20px;padding:1.5rem 1.7rem;box-shadow:0 18px 44px rgba(2,8,23,.5);}
.hero-eyebrow{font-size:.68rem;font-weight:700;letter-spacing:.16em;color:var(--muted);text-transform:uppercase;margin-bottom:.45rem;}
.hero-aqi{font-size:4.6rem;font-weight:800;line-height:1;letter-spacing:-.05em;color:var(--text);}
.hero-row{display:flex;align-items:center;gap:.65rem;margin-top:.65rem;flex-wrap:wrap;}
.hero-health{color:var(--muted);font-size:.92rem;line-height:1.55;margin-top:.7rem;max-width:38rem;}

/* ---------- Metric cards ---------- */
.metric-card{background:linear-gradient(180deg,rgba(18,28,52,.92),rgba(12,20,36,.92));border:1px solid var(--border);border-radius:16px;padding:1.05rem 1.15rem;min-height:118px;transition:border-color .15s ease,transform .15s ease;}
.metric-card:hover{border-color:#2c3f68;transform:translateY(-2px);}
.mc-top{display:flex;align-items:center;gap:.55rem;margin-bottom:.7rem;}
.mc-icon{width:32px;height:32px;border-radius:9px;display:flex;align-items:center;justify-content:center;}
.mc-icon svg{width:18px;height:18px;stroke:currentColor;fill:none;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round;}
.mc-label{font-size:.68rem;font-weight:700;letter-spacing:.11em;color:var(--muted);text-transform:uppercase;}
.mc-value{font-size:1.7rem;font-weight:800;color:var(--text);letter-spacing:-.02em;line-height:1.1;}
.mc-unit{font-size:.95rem;font-weight:600;color:var(--muted);letter-spacing:0;}
.mc-sub{font-size:.74rem;color:var(--muted-2);margin-top:.5rem;font-weight:500;}

/* ---------- Health status ---------- */
.health-card{background:linear-gradient(135deg,rgba(18,28,52,.92),rgba(12,20,36,.92));border:1px solid var(--border);border-left:4px solid var(--hc,#2dd4bf);border-radius:16px;padding:1.15rem 1.35rem;margin-top:.9rem;}
.health-title{font-size:.68rem;font-weight:700;letter-spacing:.15em;color:var(--muted);text-transform:uppercase;margin-bottom:.45rem;}
.health-cat{font-size:1.35rem;font-weight:800;color:var(--hc,#2dd4bf);letter-spacing:-.01em;}
.health-msg{font-size:.9rem;color:var(--muted);line-height:1.55;margin-top:.4rem;}

/* ---------- Day summary cards ---------- */
.day-card{background:linear-gradient(180deg,rgba(18,28,52,.92),rgba(12,20,36,.92));border:1px solid var(--border);border-radius:18px;padding:1.15rem 1.3rem;text-align:center;box-shadow:0 12px 30px rgba(2,8,23,.4);}
.day-title{margin-bottom:.2rem;}
.day-sub{font-size:.66rem;font-weight:700;letter-spacing:.14em;color:var(--muted-2);text-transform:uppercase;margin-top:.55rem;}
.day-avg{font-size:2.6rem;font-weight:800;line-height:1.05;letter-spacing:-.03em;color:var(--text);margin:.1rem 0 .3rem;}
.day-range{font-size:.78rem;color:var(--muted);margin-top:.65rem;font-weight:600;}

/* ---------- Risk panel ---------- */
.risk-card{display:flex;align-items:center;justify-content:space-between;gap:1.2rem;flex-wrap:wrap;background:linear-gradient(135deg,rgba(18,28,52,.94),rgba(12,20,36,.94));border:1px solid var(--border);border-radius:18px;padding:1.25rem 1.5rem;margin-top:.9rem;}
.risk-label{font-size:.68rem;font-weight:700;letter-spacing:.15em;color:var(--muted);text-transform:uppercase;}
.risk-peak{display:flex;align-items:baseline;gap:.55rem;margin-top:.35rem;flex-wrap:wrap;}
.risk-num{font-size:3rem;font-weight:800;letter-spacing:-.03em;line-height:1;color:var(--rk,#f87171);}
.risk-low{font-size:.82rem;color:var(--muted);margin-top:.5rem;font-weight:500;}
.risk-side{font-size:.84rem;color:var(--muted);line-height:1.7;text-align:right;}

/* ---------- Status cards ---------- */
.status-card{background:linear-gradient(180deg,rgba(18,28,52,.9),rgba(12,20,36,.9));border:1px solid var(--border);border-radius:14px;padding:.95rem 1.1rem;min-height:86px;display:flex;flex-direction:column;gap:.35rem;}
.status-label{font-size:.66rem;font-weight:700;letter-spacing:.12em;color:var(--muted);text-transform:uppercase;display:flex;align-items:center;gap:.45rem;}
.status-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.status-value{font-size:1.02rem;font-weight:700;color:var(--text);word-break:break-word;line-height:1.35;}

/* ---------- Alerts / error panels ---------- */
.panel-alert{border-radius:14px;padding:.95rem 1.2rem;margin:.35rem 0 .8rem;border:1px solid;font-size:.9rem;line-height:1.55;}
.alert-error{background:rgba(248,113,113,.08);border-color:rgba(248,113,113,.35);color:#fca5a5;}
.alert-warn{background:rgba(251,191,36,.08);border-color:rgba(251,191,36,.35);color:#fcd34d;}
.alert-info{background:rgba(56,189,248,.08);border-color:rgba(56,189,248,.35);color:#7dd3fc;}
.alert-success{background:rgba(74,222,128,.08);border-color:rgba(74,222,128,.35);color:#86efac;}
.alert-title{font-weight:800;font-size:.95rem;margin-bottom:.15rem;letter-spacing:.01em;}

/* ---------- Badges / chips ---------- */
.aqi-badge{display:inline-block;padding:.3rem .75rem;border-radius:999px;font-size:.76rem;font-weight:700;letter-spacing:.02em;line-height:1.35;}
.chip-row{display:flex;gap:.6rem;flex-wrap:wrap;margin-bottom:.8rem;}
.chip{display:inline-flex;align-items:center;gap:.5rem;background:rgba(16,26,48,.85);border:1px solid var(--border);border-radius:10px;padding:.5rem .85rem;font-size:.78rem;font-weight:600;color:var(--muted);}
.chip b{color:var(--text);font-weight:800;}
.chip .dot{width:7px;height:7px;border-radius:50%;background:var(--accent);flex-shrink:0;}

/* ---------- Sidebar mini-status ---------- */
.side-brand{font-weight:800;font-size:1rem;color:var(--text);letter-spacing:-.01em;}
.side-sub{color:var(--muted);font-size:.72rem;margin:.15rem 0 .6rem;}
.side-kicker{font-size:.64rem;font-weight:700;letter-spacing:.14em;color:var(--muted);text-transform:uppercase;margin:.95rem 0 .4rem;}
.side-row{display:flex;justify-content:space-between;align-items:center;gap:.6rem;font-size:.8rem;color:var(--muted);margin:.3rem 0;font-weight:500;}
.side-val{color:var(--text);font-weight:700;text-align:right;font-size:.8rem;}
.ok-tag{color:var(--ok);font-weight:700;font-size:.76rem;display:inline-flex;align-items:center;gap:.35rem;}
.ok-tag i{width:7px;height:7px;border-radius:50%;background:var(--ok);display:inline-block;animation:pulse 2s infinite;}

/* ---------- Map / footer ---------- */
.map-wrap{background:linear-gradient(180deg,rgba(18,28,52,.7),rgba(12,20,36,.7));border:1px solid var(--border);border-radius:16px;padding:.9rem;}
.app-footer{margin:2.6rem 0 .6rem;border-top:1px solid var(--border);padding-top:1.15rem;text-align:center;}
.foot-brand{font-weight:800;color:var(--text);font-size:.95rem;letter-spacing:.02em;}
.foot-line{color:var(--muted-2);font-size:.78rem;margin-top:.35rem;line-height:1.75;}
.foot-line code{color:var(--accent-2);background:rgba(34,211,238,.08);padding:.1rem .35rem;border-radius:5px;font-size:.74rem;}
</style>""", unsafe_allow_html=True)



_ICON_AQI = '<svg viewBox="0 0 24 24"><path d="M4 14a8 8 0 1 1 16 0"/><path d="M12 14l4-4"/><circle cx="12" cy="14" r="1.7"/></svg>'
_ICON_PM = '<svg viewBox="0 0 24 24"><circle cx="8" cy="7" r="1.8"/><circle cx="16" cy="10" r="1.4"/><circle cx="12" cy="16" r="2"/><circle cx="18" cy="18" r="1.2"/></svg>'
_ICON_PM10 = '<svg viewBox="0 0 24 24"><circle cx="9" cy="10" r="3"/><circle cx="16" cy="15" r="3"/><path d="M6 17h4"/></svg>'
_ICON_TEMP = '<svg viewBox="0 0 24 24"><path d="M10 4a2 2 0 1 1 4 0v9.2a4 4 0 1 1-4 0z"/><circle cx="12" cy="17" r="1.7"/></svg>'
_ICON_HUM = '<svg viewBox="0 0 24 24"><path d="M12 3.5S6 10 6 14.5a6 6 0 0 0 12 0C18 10 12 3.5 12 3.5z"/></svg>'
_ICON_WIND = '<svg viewBox="0 0 24 24"><path d="M3 8h11a3 3 0 1 0-3-3"/><path d="M3 12h15a3 3 0 1 1-3 3"/><path d="M3 16h7"/></svg>'
_ICON_PIN = '<svg viewBox="0 0 24 24"><path d="M12 21s-7-5.1-7-11a7 7 0 0 1 14 0c0 5.9-7 11-7 11z"/><circle cx="12" cy="10" r="2.6"/></svg>'
_ICON_CLOCK = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.2"/><path d="M12 7.5V12l3 2"/></svg>'
_ICON_AIR = '<svg viewBox="0 0 24 24"><path d="M3 8c3-2.5 6-2.5 9 0s6 2.5 9 0"/><path d="M3 12c3-2.5 6-2.5 9 0s6 2.5 9 0"/><path d="M3 16c3-2.5 6-2.5 9 0s6 2.5 9 0"/></svg>'


def _aqi_hex(category, default="#8fa3c6"):
    """Presentation helper: normalized hex color for an AQI category (from AQI_CATEGORY_COLORS)."""
    c = AQI_CATEGORY_COLORS.get(category)
    return c if isinstance(c, str) and c.startswith("#") and len(c) == 7 else default


def _aqi_badge(category):
    """Presentation helper: colored pill showing an AQI category (uses AQI_CATEGORY_COLORS)."""
    c = _aqi_hex(category)
    return (f'<span class="aqi-badge" style="color:{c};background:{c}26;'
            f'border:1px solid {c}59;">{category}</span>')


def _now_stamp(fmt="%I:%M %p"):
    return datetime.now().strftime(fmt)


def _fmt_ts(ts):
    """Compact display of a timestamp (real value, not hardcoded)."""
    if hasattr(ts, "strftime"):
        try:
            return ts.strftime("%b %d, %H:%M")
        except Exception:
            pass
    return str(ts)


def _chart_theme(fig, height=380):
    """Reusable dark Plotly theme — presentation only, data untouched."""
    fig.update_layout(
        template="plotly_dark", height=height, hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color="#e8eef8", size=12.5),
        xaxis=dict(gridcolor="rgba(148,163,184,0.10)", zeroline=False,
                   linecolor="rgba(148,163,184,0.18)", tickfont=dict(color="#8fa3c6")),
        yaxis=dict(gridcolor="rgba(148,163,184,0.10)", zeroline=False,
                   linecolor="rgba(148,163,184,0.18)", tickfont=dict(color="#8fa3c6")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")),
        margin=dict(l=16, r=16, t=24, b=16),
        hoverlabel=dict(bgcolor="#16213c", bordercolor="#2b3a5c", font=dict(color="#e8eef8")),
    )
    return fig


def render_header():
    st.markdown(
        '<div class="app-header">'
        '<div class="brand">'
        f'<div class="brand-mark">{_ICON_AIR}</div>'
        '<div><div class="brand-title">KARACHI AIR INTELLIGENCE</div>'
        '<div class="brand-sub">AI-Powered 72-Hour Air Quality Forecasting</div></div>'
        '</div>'
        '<div class="header-meta">'
        '<div class="live-pill"><i></i>SYSTEM ONLINE</div>'
        '<div class="meta-row">'
        f'<span class="meta-pill">{_ICON_PIN} Karachi, Pakistan</span>'
        f'<span class="meta-pill">{_ICON_CLOCK} Updated {_now_stamp()}</span>'
        '</div></div></div>',
        unsafe_allow_html=True,
    )


def render_section_header(kicker, title, subtitle=None):
    sub = f'<div class="sec-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="sec-head"><div class="sec-bar"></div><div>'
        f'<div class="sec-kicker">{kicker}</div>'
        f'<div class="sec-title">{title}</div>{sub}</div></div>',
        unsafe_allow_html=True,
    )


def render_pipeline_arch():
    st.markdown(
        '<div class="pipe-wrap"><div class="pipeline">'
        '<div class="pipe-node">LIVE DATA · Open-Meteo</div><div class="pipe-arrow">→</div>'
        '<div class="pipe-node">FEATURE ENGINEERING</div><div class="pipe-arrow">→</div>'
        '<div class="pipe-node">FEAST FEATURE STORE</div><div class="pipe-arrow">→</div>'
        '<div class="pipe-node">ONLINE RETRIEVAL</div><div class="pipe-arrow">→</div>'
        '<div class="pipe-node">ML MODEL</div><div class="pipe-arrow">→</div>'
        '<div class="pipe-node hot">72H FORECAST</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def render_metric_card(icon, label, value, unit, sub, color="#2dd4bf"):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="mc-top"><span class="mc-icon" style="color:{color};background:{color}14;">{icon}</span>'
        f'<span class="mc-label">{label}</span></div>'
        f'<div class="mc-value">{value}<span class="mc-unit"> {unit}</span></div>'
        f'<div class="mc-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def render_alert(kind, title, body=None):
    """Styled alert panel. kind: error | warn | info | success. Keeps message text visible."""
    body_html = f'<div>{html.escape(body)}</div>' if body else ""
    st.markdown(
        f'<div class="panel-alert alert-{kind}"><div class="alert-title">{html.escape(title)}</div>{body_html}</div>',
        unsafe_allow_html=True,
    )


def render_aqi_gauge(aqi, category):
    c = _aqi_hex(category)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi,
        number={"suffix": "", "font": {"size": 34, "color": c, "family": "Inter, sans-serif"}},
        title={"text": "US AQI", "font": {"size": 12, "color": "#8fa3c6", "family": "Inter, sans-serif"}},
        gauge={
            "axis": {"range": [0, 500], "tickcolor": "#8fa3c6", "tickfont": {"color": "#8fa3c6", "size": 10}},
            "bar": {"color": c, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(74,222,128,0.12)"},
                {"range": [50, 100], "color": "rgba(250,204,21,0.12)"},
                {"range": [100, 150], "color": "rgba(251,146,60,0.12)"},
                {"range": [150, 200], "color": "rgba(248,113,113,0.12)"},
                {"range": [200, 300], "color": "rgba(167,139,250,0.12)"},
                {"range": [300, 500], "color": "rgba(190,18,60,0.12)"},
            ],
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=215,
                      font=dict(family="Inter, sans-serif", color="#8fa3c6"),
                      margin=dict(l=12, r=12, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def render_health_card(aqi, category):
    c = _aqi_hex(category)
    st.markdown(
        f'<div class="health-card" style="--hc:{c};">'
        f'<div class="health-title">Air Quality Status</div>'
        f'<div class="health-cat">{category}</div>'
        f'<div class="health-msg">{html.escape(aqi_health_message(aqi))}</div></div>',
        unsafe_allow_html=True,
    )


def render_day_cards(day_stats):
    cols = st.columns(3)
    for i, (day, row) in enumerate(day_stats.iterrows()):
        if i >= 3:
            break
        cat = aqi_to_category(row["mean"])
        c = _aqi_hex(cat)
        with cols[i]:
            st.markdown(
                f'<div class="day-card">'
                f'<div class="day-title">{_aqi_badge(day)}</div>'
                f'<div class="day-sub">Average AQI</div>'
                f'<div class="day-avg" style="color:{c};">{row["mean"]:.0f}</div>'
                f'{_aqi_badge(cat)}'
                f'<div class="day-range">Min {row["min"]:.0f} &nbsp;·&nbsp; Max {row["max"]:.0f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_risk_panel(forecast_df):
    overall_max = forecast_df["predicted_aqi"].max()
    overall_min = forecast_df["predicted_aqi"].min()
    peak_idx = forecast_df["predicted_aqi"].idxmax()
    peak_cat = aqi_to_category(overall_max)
    min_cat = aqi_to_category(overall_min)
    cmax = _aqi_hex(peak_cat)
    peak_ts = forecast_df.loc[peak_idx, "timestamp"]
    try:
        peak_label = peak_ts.strftime("%a %b %d · %I:%M %p")
    except Exception:
        peak_label = str(peak_ts)
    peak_day = forecast_df.loc[peak_idx, "forecast_day"]
    st.markdown(
        f'<div class="risk-card" style="--rk:{cmax};">'
        f'<div><div class="risk-label">Forecast Risk · Next 72 Hours</div>'
        f'<div class="risk-peak"><span class="risk-num">{overall_max:.0f}</span>{_aqi_badge(peak_cat)}</div>'
        f'<div class="risk-low">Highest forecasted AQI &nbsp;·&nbsp; Low point '
        f'<b style="color:var(--text);">{overall_min:.0f}</b> ({min_cat})</div></div>'
        f'<div class="risk-side">'
        f'<div>Peak expected around <b style="color:var(--text);">{peak_label}</b></div>'
        f'<div>{peak_day} of the forecast window</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_status_card(label, value, ok=True, dot_color=None):
    dot = dot_color or (("#4ade80" if ok else "#fbbf24"))
    st.markdown(
        f'<div class="status-card">'
        f'<div class="status-label"><span class="status-dot" style="background:{dot};"></span>{label}</div>'
        f'<div class="status-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def render_footer(model_name):
    st.markdown(
        '<div class="app-footer">'
        '<div class="foot-brand">KARACHI AIR INTELLIGENCE</div>'
        f'<div class="foot-line">AI-powered 72-hour US AQI forecasting &nbsp;·&nbsp; '
        f'updated {_now_stamp("%Y-%m-%d %H:%M")}</div>'
        f'<div class="foot-line">Model: <code>{html.escape(model_name)}</code> '
        f'&nbsp;·&nbsp; Feature Store: <code>Feast</code> '
        f'&nbsp;·&nbsp; Forecast Horizon: <code>{FORECAST_HOURS} hours</code> '
        f'&nbsp;·&nbsp; Data Source: <code>Open-Meteo</code> '
        f'&nbsp;·&nbsp; Location: <code>{CITY_NAME}, Pakistan</code></div>'
        '</div>',
        unsafe_allow_html=True,
    )



render_header()
render_pipeline_arch()

artifacts = load_production_model()
if artifacts is None:
    render_alert(
        "error", "Model not found",
        "No trained model found in `models/`. Run the notebook "
        "`notebooks/Karachi_AQI_Forecasting.ipynb` first (through Section 24, 'Save Model and Outputs') "
        "so that `models/model_config.json` and the model files exist.",
    )
    st.stop()

feature_cols = artifacts["feature_cols"]


setup_check = fu.verify_feast_setup()

if not setup_check["ok"]:
    with st.spinner("Feast online store has no data yet - materializing from your existing feature history..."):
        attempted, setup_check = fu.bootstrap_materialize_if_needed()
    if attempted and setup_check["ok"]:
        st.success("Feast online store was empty - automatically materialized from your existing offline "
                    "feature history. Continuing normally.")

if not setup_check["ok"]:
    render_alert("error", "Feast Feature Store setup is incomplete",
                 "The dashboard cannot retrieve online features yet.")
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
    render_alert("info", "Recovery",
                 "Run the commands above (e.g. in a notebook cell or a Python shell in this project's "
                 "environment), then reload this page.")
    st.stop()

online_df = fu.get_latest_features_from_feast()
feast_connected = True


with st.spinner("Fetching live weather & air quality data from Open-Meteo..."):
    try:
        weather_df, air_df = fetch_recent_and_forecast()
        fetch_error = None
    except Exception as e:
        fetch_error = str(e)
        weather_df = air_df = None

if fetch_error:
    render_alert("error", "Open-Meteo data source unavailable", fetch_error)
    if not feast_connected:
        st.stop()
    render_alert("info", "Fallback active",
                 "Falling back to the last materialized Feast online features (no fresh observation pushed this run).")
    weather_history = None
    future_weather = None
else:
    with st.spinner("Updating Feast with the latest observation..."):
        try:
            weather_history, future_weather = update_feast_with_latest_observations(weather_df, air_df, feature_cols)
            
            online_df = fu.get_latest_features_from_feast()
            feast_connected = True
        except Exception as e:
            render_alert("warn", "Feast refresh deferred",
                         f"Could not update Feast with a fresh observation this run ({e}). "
                         f"Using whatever was last materialized.")
            weather_history = None
            future_weather = None

if not feast_connected or online_df is None:
    render_alert(
        "error", "No online features available",
        "Feast online store has no features available yet. In the notebook, run through "
        "'Materialize Features' at least once (Feast Feature Store Setup section) before "
        "using this dashboard.",
    )
    st.stop()


with st.sidebar:
    st.markdown('<div class="side-brand">KARACHI AIR INTELLIGENCE</div>'
                '<div class="side-sub">AI-Powered 72-Hour AQI Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="ok-tag"><i></i>SYSTEM ONLINE</div>', unsafe_allow_html=True)
    st.caption(f"Dashboard updated {_now_stamp()}")

    st.markdown('<div class="side-kicker">System Status</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-row"><span>Feature Store</span><span class="side-val" style="color:#4ade80;">● Connected</span></div>',
                unsafe_allow_html=True)
    ts_val = online_df["event_timestamp"].iloc[0] if "event_timestamp" in online_df.columns else "n/a"
    

    model_name = artifacts["config"]["final_model_name"]
    aqi_standard = artifacts["config"]["aqi_standard"]
    st.markdown(f'<div class="side-row"><span>ML Model</span><span class="side-val">{html.escape(model_name)}</span></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="side-row"><span>AQI Standard</span><span class="side-val">{html.escape(aqi_standard)}</span></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="side-row"><span>Horizon</span><span class="side-val">{FORECAST_HOURS} hours</span></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="side-row"><span>Location</span><span class="side-val">{CITY_NAME}, Pakistan</span></div>',
                unsafe_allow_html=True)

    st.divider()
    # st.caption(f"Feast feature service: `{fu.FEATURE_SERVICE_NAME}` · {len(feature_cols)} features · "
    #            f"({LATITUDE}, {LONGITUDE})")

# ---------------- Online feature vector for the first prediction step (UNCHANGED) ----------------
try:
    online_vector = fu.get_online_features_for_prediction(feature_cols)
except fu.FeastConfigError as e:
    render_alert("error", "Feast online retrieval failed", str(e))
    render_alert("info", "Diagnostics",
                 "Run `fu.verify_feast_setup()` for a full diagnostic, or the recovery commands "
                 "shown in the Feature Store panel above.")
    st.stop()

# ---------------- Forecast (UNCHANGED) ----------------
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

# ==========================================================================
# Main dashboard sections (presentation organization only)
# ==========================================================================
current_aqi = float(online_df["us_aqi_lag_1"].iloc[0]) if "us_aqi_lag_1" in online_df.columns else None

# ---------- 1) HERO / CURRENT AQI ----------
render_section_header("Live Conditions", "Current Air Quality",
                    
                      )

if current_aqi is not None:
    cat = aqi_to_category(current_aqi)
    c = _aqi_hex(cat)
    left, right = st.columns([2.0, 1.0])
    with left:
        st.markdown(
            f'<div class="hero-card" style="--heroacc:{c};">'
            f'<div><div class="hero-eyebrow">Current Air Quality</div>'
            f'<div class="hero-aqi">{current_aqi:.0f}</div>'
            f'<div class="hero-row">{_aqi_badge(cat)}'
            f'<span class="meta-pill">US AQI · {_fmt_ts(ts_val)}</span></div>'
            f'<div class="hero-health">{html.escape(aqi_health_message(current_aqi))}</div></div></div>',
            unsafe_allow_html=True,
        )
    with right:
        render_aqi_gauge(current_aqi, cat)
    
    render_health_card(current_aqi, cat)
else:
    render_alert("warn", "Current AQI unavailable",
                 "No current AQI available from the feature store.")

# ---------- 2) CURRENT CONDITIONS ----------
render_section_header("Live Monitoring", "Current Conditions",
                      "Latest pollutant concentrations & meteorology")

pm25 = online_df["pm2_5_lag_1"].iloc[0] if "pm2_5_lag_1" in online_df.columns else None
pm10 = online_df["pm10_lag_1"].iloc[0] if "pm10_lag_1" in online_df.columns else None
temp = online_df["temperature_2m"].iloc[0] if "temperature_2m" in online_df.columns else None
hum = online_df["relative_humidity_2m"].iloc[0] if "relative_humidity_2m" in online_df.columns else None
wind = online_df["wind_speed_10m"].iloc[0] if "wind_speed_10m" in online_df.columns else None

r1a, r1b, r1c = st.columns(3)
with r1a:
    render_metric_card(_ICON_AQI, "Air Quality Index",
                       f"{current_aqi:.0f}" if current_aqi is not None else "n/a", "US AQI",
                       aqi_to_category(current_aqi) if current_aqi is not None else "not available",
                       color=_aqi_hex(aqi_to_category(current_aqi)) if current_aqi is not None else "#8fa3c6")
with r1b:
    render_metric_card(_ICON_PM, "PM2.5", f"{pm25:.1f}" if pm25 is not None else "n/a", "µg/m³",
                       "Current concentration", color="#22d3ee")
with r1c:
    render_metric_card(_ICON_PM10, "PM10", f"{pm10:.1f}" if pm10 is not None else "n/a", "µg/m³",
                       "Current concentration", color="#a78bfa")

r2a, r2b, r2c = st.columns(3)
with r2a:
    render_metric_card(_ICON_TEMP, "Temperature", f"{temp:.1f}" if temp is not None else "n/a", "°C",
                       "Open-Meteo observation", color="#fbbf24")
with r2b:
    render_metric_card(_ICON_HUM, "Humidity", f"{hum:.0f}" if hum is not None else "n/a", "%",
                       "Relative humidity", color="#34d399")
with r2c:
    render_metric_card(_ICON_WIND, "Wind Speed", f"{wind:.1f}" if wind is not None else "n/a", "km/h",
                       "10m wind speed", color="#38bdf8")

# ---------- 3) 72-HOUR AI FORECAST ----------
render_section_header("Forecast", "72-Hour AI Forecast",
                      "Hourly AQI prediction powered by Machine Learning")

if forecast_error:
    render_alert("warn", "Forecast pipeline unavailable", forecast_error)
elif forecast_df is not None:
    fig = go.Figure()
    if weather_history is not None:
        obs = weather_history.tail(48)
        fig.add_trace(go.Scatter(x=obs["timestamp"], y=obs["us_aqi"], mode="lines", name="Observed",
                                 line=dict(color="#60a5fa", width=2.4),
                                 hovertemplate="%{x|%a %d %b, %H:%M}<br>Observed AQI: %{y:.1f}<extra></extra>"))

    fc_custom = np.stack([forecast_df["forecast_hour"], forecast_df["aqi_category"]], axis=-1)
    fig.add_trace(go.Scatter(x=forecast_df["timestamp"], y=forecast_df["predicted_aqi"],
                             mode="lines", name="AI Forecast",
                             line=dict(color="#2dd4bf", width=2.6),
                             customdata=fc_custom,
                             hovertemplate="%{x|%a %d %b, %H:%M}<br>Forecast AQI: %{y:.1f}"
                                           "<br>Category: %{customdata[1]}<extra></extra>"))

    sev_colors = [_aqi_hex(cat) for cat in forecast_df["aqi_category"]]
    fig.add_trace(go.Scatter(x=forecast_df["timestamp"], y=forecast_df["predicted_aqi"],
                             mode="markers", name="Severity",
                             marker=dict(size=5.5, color=sev_colors,
                                         line=dict(width=0.6, color="rgba(10,15,28,0.9)")),
                             hoverinfo="skip", showlegend=False))

    if weather_history is not None:
        t0 = weather_history["timestamp"].max()
        fig.add_vline(x=t0, line=dict(color="#8fa3c6", width=1, dash="dot"),
                      annotation_text="NOW", annotation_position="top right",
                      annotation_font=dict(color="#8fa3c6", size=11))

    _chart_theme(fig, height=470)
    fig.update_layout(xaxis_title="", yaxis_title="US AQI",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


    day_stats = forecast_df.groupby("forecast_day")["predicted_aqi"].agg(["mean", "min", "max"])
    render_day_cards(day_stats)

  
    render_risk_panel(forecast_df)

    
    with st.expander("Hourly Forecast Details — full 72-hour table", expanded=False):
        st.dataframe(
            forecast_df[["timestamp", "predicted_aqi", "aqi_category", "forecast_day", "forecast_hour"]],
            use_container_width=True, hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
                "predicted_aqi": st.column_config.NumberColumn("Predicted AQI", format="%.1f"),
                "aqi_category": "AQI Category",
                "forecast_day": "Forecast Day",
                "forecast_hour": "Hour",
            },
        )

# ---------- 4) POLLUTANT & WEATHER ANALYTICS ----------
if forecast_df is not None:
    render_section_header("Analytics", "Pollutant & Weather Analytics",
                          "Recent 72 hours — observed pollutant concentrations and meteorology")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown('<div class="sec-sub" style="margin:0 0 .35rem;font-weight:700;">POLLUTANT TRENDS</div>',
                    unsafe_allow_html=True)
        fig_pm = px.line(weather_history.tail(72), x="timestamp", y=["pm2_5", "pm10"],
                         labels={"value": "µg/m³", "timestamp": "",
                                 "pm2_5": "PM2.5 (µg/m³)", "pm10": "PM10 (µg/m³)", "variable": ""},
                         color_discrete_sequence=["#38bdf8", "#a78bfa"])
        _chart_theme(fig_pm, 330)
        fig_pm.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig_pm, use_container_width=True, config={"displaylogo": False})
    with t2:
        st.markdown('<div class="sec-sub" style="margin:0 0 .35rem;font-weight:700;">WEATHER CONDITIONS</div>',
                    unsafe_allow_html=True)
        w = weather_history.tail(72)
        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(x=w["timestamp"], y=w["temperature_2m"], name="Temperature (°C)",
                                   line=dict(color="#fbbf24", width=2.2),
                                   hovertemplate="%{x|%a %d %b, %H:%M}<br>Temperature: %{y:.1f} °C<extra></extra>"))
        fig_w.add_trace(go.Scatter(x=w["timestamp"], y=w["relative_humidity_2m"], name="Humidity (%)",
                                   yaxis="y2", line=dict(color="#34d399", width=2.2),
                                   hovertemplate="%{x|%a %d %b, %H:%M}<br>Humidity: %{y:.0f} %<extra></extra>"))
        _chart_theme(fig_w, 330)
        fig_w.update_layout(
            yaxis=dict(title="Temperature (°C)", gridcolor="rgba(148,163,184,0.10)",
                       zeroline=False, linecolor="rgba(148,163,184,0.18)", tickfont=dict(color="#8fa3c6")),
            yaxis2=dict(title="Humidity (%)", overlaying="y", side="right", showgrid=False,
                        linecolor="rgba(148,163,184,0.18)", tickfont=dict(color="#8fa3c6")),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig_w, use_container_width=True, config={"displaylogo": False})
else:
    render_section_header("Analytics", "Pollutant & Weather Analytics",
                          "Recent 72 hours — observed pollutant concentrations and meteorology")
    render_alert("info", "Trends unavailable",
                 "Forecast pipeline unavailable this run (no live weather data) - trends cannot be rendered.")

# ---------- 5) AI SYSTEM STATUS ----------
render_section_header("ML Architecture", "AI System Status",
                      "Production pipeline health — feature store, model and serving configuration")

s1a, s1b, s1c = st.columns(3)
with s1a:
    render_status_card("Feature Store", "Connected", ok=True)
with s1b:
    render_status_card("Online Features", f"Latest: {_fmt_ts(ts_val)}", ok=True)
with s1c:
    render_status_card("ML Model", model_name, ok=True)

s2a, s2b, s2c = st.columns(3)
with s2a:
    render_status_card("Forecast Horizon", f"{FORECAST_HOURS} Hours", ok=True)
with s2b:
    render_status_card("AQI Standard", aqi_standard, ok=True)
with s2c:
    render_status_card("Data Source", "Open-Meteo", ok=True)

# ---------- 6) FEATURE STORE INSIGHTS ----------
render_section_header("Feature Store", "Feature Store Insights",
                      "Live online retrieval — the prediction pipeline is served directly from Feast")

st.markdown(
    '<div class="chip-row">'
    f'<span class="chip"><span class="dot"></span><b>{len(feature_cols)}</b>&nbsp;features available</span>'
    f'<span class="chip"><span class="dot"></span>Service: <b>&nbsp;{fu.FEATURE_SERVICE_NAME}</b></span>'
    '<span class="chip"><span class="dot"></span>Source: <b>&nbsp;Feast Online Store</b></span>'
    '</div>',
    unsafe_allow_html=True,
)

with st.expander("Features Retrieved from Feast", expanded=True):
    display_feats = ["us_aqi_lag_1", "us_aqi_rolling_mean_24", "temperature_2m",
                      "relative_humidity_2m", "surface_pressure", "wind_speed_10m"]
    display_feats = [c for c in display_feats if c in online_df.columns]
    table_rows = [{"Feature Name": c, "Value": round(float(online_df[c].iloc[0]), 2)} for c in display_feats]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True,
                 column_config={
                     "Feature Name": "Feature Name",
                     "Value": st.column_config.NumberColumn("Latest Value", format="%.2f"),
                 })
    st.caption(f"{len(feature_cols)} total features available via the '{fu.FEATURE_SERVICE_NAME}' feature service "
               f"(showing a representative subset above).")

# ---------- 7) MONITORING LOCATION ----------
render_section_header("Coverage", "Monitoring Location",
                      f"{CITY_NAME}, Pakistan — live sensor anchor ({LATITUDE}, {LONGITUDE})")

st.markdown('<div class="map-wrap">', unsafe_allow_html=True)
if HAS_FOLIUM:
    m = folium.Map(location=[LATITUDE, LONGITUDE], zoom_start=11)
    folium.Marker([LATITUDE, LONGITUDE], popup=f"{CITY_NAME} ({LATITUDE}, {LONGITUDE})",
                  tooltip="Forecast location").add_to(m)
    st_folium(m, width=700, height=350)
else:
    st.map(pd.DataFrame({"lat": [LATITUDE], "lon": [LONGITUDE]}))
    st.caption("Install `folium` and `streamlit-folium` for a richer map (`pip install folium streamlit-folium`).")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- 8) TECHNICAL FOOTER ----------
render_footer(model_name)
