# Karachi AQI Forecasting

An end-to-end machine learning project that predicts Karachi's Air Quality Index (AQI) for the next 3 days, hour by hour.

Built during my internship, this project takes historical weather and air-quality data, runs it through a full ML pipeline, and outputs a 72-hour AQI forecast on a live dashboard — all automated with CI/CD.

---

## What it does

Karachi's air quality changes fast, and most people only find out it's bad *after* they're already outside in it. This project tries to fix that by forecasting AQI a full three days ahead, updated hour by hour, so the numbers are actually useful for planning.

**Highlights:**
- 72-hour AQI forecast, broken down hour by hour
- Historical AQI trend analysis
- Automated data collection and feature engineering
- A proper Feature Store (Feast) instead of ad-hoc CSVs
- Several ML models trained and compared, with the best one picked automatically
- CI/CD pipelines that keep data, features, and models fresh without manual work
- A clean, interactive Streamlit dashboard

---

## How it all fits together

```
Data Collection → Cleaning → EDA → Feature Engineering → Feast Feature Store
   → Model Training → Model Evaluation → Best Model Selection
   → CI/CD → Streamlit Dashboard → 72-Hour Forecast
```

---

## The data

All data comes from [Open-Meteo](https://open-meteo.com/), covering **September 2022 to July 2026**.

**Air quality:** AQI, PM2.5, PM10, CO, NO2, SO2, O3
**Weather:** temperature, humidity, pressure, wind, and a few other variables

For live forecasts, the same data is pulled fresh from the Open-Meteo API.

---

## Exploring the data

Before touching any models, I dug into the data to understand what I was working with — missing values, duplicates, data types, how AQI is distributed, how it moves over time, and how it correlates with weather variables. This step shaped a lot of the feature engineering decisions later on.

---

## Cleaning and preparing the data

Since this is a time-series forecasting problem, keeping everything in chronological order mattered a lot. The prep work involved sorting by timestamp, handling missing values, cleaning up inconsistent records, and getting timestamps into a usable format.

---

## Feature engineering

This is where most of the real work happened. A few types of features made the biggest difference:

**Lag features** — AQI from the previous hour, previous few hours, and the previous day, since air quality doesn't change randomly; it trends.

**Rolling features** — rolling mean and standard deviation, to capture recent behavior rather than just single points in time.

**Weather change features** — how temperature, humidity, and pressure are shifting, since those changes often precede AQI shifts.

Together, these give the model a sense of *momentum* — not just where AQI is right now, but where it's heading.

---

## What's being forecasted

```
Day 1 → 24 hourly predictions
Day 2 → 24 hourly predictions
Day 3 → 24 hourly predictions
Total → 72 hourly predictions
```

---

## Models tried

I tested a range of models rather than betting on one from the start:

Ridge Regression, Naive Persistence, Seasonal Persistence, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, LightGBM, XGBoost, CatBoost, and an LSTM.

Each was evaluated using MAE, RMSE, and R².

## Model Selection

Based on the evaluation of all trained models, a weighted ensemble blending approach was selected as the final forecasting model. The ensemble combines the predictions of the Tuned LightGBM, Tuned XGBoost, and Extra Trees models.
---

## Why Feast for the Feature Store

I wanted to keep feature generation and model prediction as separate concerns instead of recomputing everything inline every time — that's exactly what a feature store is for. Feast handles:

- Storing and versioning features
- Managing feature definitions
- Serving the latest features at prediction time
- Keeping training and serving features consistent

**Setup:**
```
Offline Store → DuckDB
Online Store  → SQLite
Registry      → Local Registry
```
DuckDB made sense for the offline store since the dataset was small enough to process locally without needing a cloud warehouse.

### Why not Hopsworks?

Hopsworks was the original plan, but its cloud dependency caused enough friction during setup that it slowed the project down. Feast, running entirely locally, was a much better fit for an internship-scale project — easier to develop against, easier to test, and with no external service to worry about.

---

## Training pipeline

The training pipeline handles the full loop automatically: load prepared data → engineer features → train models → evaluate them → save the best one → update project files. The dashboard then reads directly from whatever the pipeline produces.

---

## CI/CD

GitHub Actions runs the project on autopilot through three workflows:

- **Feature pipeline** — pulls new data and refreshes the feature store on a schedule
- **Training pipeline** — retrains and re-evaluates models regularly
- **CI validation** — checks that the project setup is actually working

```
New Data → Feature Pipeline → Feature Store → Training Pipeline → New Model → Dashboard
```

This keeps the whole system self-updating instead of needing manual reruns every time new data comes in.

---

## Dashboard

A Streamlit app ties everything together — current AQI, the full 72-hour forecast, hour-by-hour breakdowns, and trend visualizations — so anyone can check the forecast without touching the code.

```
Latest Data → Feature Engineering → Feast → Trained Model → 72-Hour Forecast → Dashboard
```

---

## Tech stack

| Category | Tools |
|---|---|
| Language | Python |
| Data & ML | Pandas, NumPy, Scikit-learn, LightGBM, XGBoost, CatBoost, TensorFlow/Keras |
| Feature Store | Feast, DuckDB, SQLite |
| Data Source | Open-Meteo API |
| Dashboard | Streamlit |
| Automation | GitHub Actions, CI/CD |
| Version Control | Git, GitHub, Git LFS |

---

## Project structure

```
Karachi-Air-Intelligence/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── feature_store/
│   ├── data/
│   └── feature_store.yaml
│
├── models/
│   ├── LightGBM_Tuned.pkl
│   ├── XGBoost_Tuned.pkl
│   ├── Extra_Trees.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── Karachi_AQI_Forecasting.ipynb
│
├── scripts/
│   ├── feature_pipeline.py
│   ├── train_pipeline.py
│   └── verify_setup.py
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml
│       ├── training_pipeline.yml
│       └── ci_validation.yml
│
└── README.md
```

---

## Results

- 3-day, 72-hour AQI forecasts, hour by hour
- Fully automated feature generation via a real feature store
- Multiple models trained, compared, and evaluated fairly
- Self-updating pipelines through CI/CD
- A live, interactive dashboard anyone can use

---

## Live app

Streamlit link: 
https://karachi-aqi-prediction-app.streamlit.app

---

## Summary

Karachi Air Intelligence is an end-to-end ML project built around one goal: making Karachi's air quality forecast genuinely useful, three days out, updated automatically, and easy to check at a glance.

**Pipeline:** Data Collection → EDA → Feature Engineering → Feast → Machine Learning → Model Evaluation → CI/CD → Dashboard