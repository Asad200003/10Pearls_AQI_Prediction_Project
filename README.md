# Karachi AQI Forecasting Internship Project

---

## 1. Project Overview

**Karachi AQI Forecasting System is an AI-powered system that forecasts Karachi's ****Air Quality Index (AQI) for the next 3 days, hour by hour**.

The system uses historical air-quality and weather data to train Machine Learning models. It then uses the latest available data to predict AQI for the next **72 hours**.

### Main Goal

The main goal of this project is to provide an easy way to understand the expected air quality in Karachi for the next three days.

### Key Features

* 72-hour AQI forecasting
* Hour-by-hour predictions
* Historical AQI analysis
* Weather and air-quality data collection
* Feature engineering
* Multiple Machine Learning models
* Feast Feature Store
* Automated CI/CD pipelines
* Interactive Streamlit dashboard
* Model training and feature pipelines

---

## 2. Project Workflow

The complete system follows this workflow:

```text
Data Collection
      ↓
Data Cleaning
      ↓
EDA
      ↓
Feature Engineering
      ↓
Feast Feature Store
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Best Model Selection
      ↓
CI/CD Pipeline
      ↓
Streamlit Dashboard
      ↓
72-Hour AQI Forecast
```

---

## 3. Data Collection

The project collects historical **air-quality and weather data for Karachi from  **[Open-Meteo.com](https://open-meteo.com/).

The data includes information such as:

### Air Quality Data

* AQI
* PM2.5
* PM10
* CO
* NO2
* SO2
* O3

### Weather Data

* Temperature
* Relative humidity
* Atmospheric pressure
* Wind-related measurements
* Other available weather variables

The historical dataset covers data from **September 2022 to July 2026**.

For the live forecasting system, the latest weather and air-quality data is collected through the **Open-Meteo API**.

---

## 4. Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset before training the models.

The analysis included:

* Checking missing values
* Checking duplicate records
* Understanding data types
* Analyzing AQI distribution
* Studying AQI over time
* Finding relationships between AQI and weather variables
* Checking correlations between features
* Identifying unusual values and patterns
* Understanding hourly and daily AQI changes

EDA helped in understanding which variables could be useful for forecasting AQI.

---

## 5. Data Preprocessing

Before model training, the data was prepared by:

* Sorting data by time
* Handling missing values
* Removing unnecessary data
* Converting timestamps correctly
* Checking data consistency
* Preparing the data for time-series forecasting

The data was kept in time order because this is a forecasting problem.

---

## 6. Feature Engineering

Feature engineering was an important part of the project.

New features were created from historical AQI and weather data to help the models understand recent patterns.

### Main Features

**Lag Features**

Previous AQI values were used, such as:

* Previous hour AQI
* Previous several hours AQI
* Previous day AQI

**Rolling Features**

Rolling statistics were created to understand recent AQI behavior:

* Rolling mean
* Rolling standard deviation

**Weather Change Features**

Changes in weather variables were also calculated, such as:

* Temperature change
* Humidity change
* Pressure change

These features help the model understand how recent conditions can affect future AQI.

---

## 7. Forecasting Target

The system is designed to forecast AQI for the next **72 hours**.

```text
Day 1 → 24 hourly AQI predictions
Day 2 → 24 hourly AQI predictions
Day 3 → 24 hourly AQI predictions

Total → 72 hourly predictions
```

The model predicts future AQI based on the latest available features and historical patterns.

---

## 8. Machine Learning Models

Multiple Machine Learning models were tested and compared.

The project included:

* Ridge Regression
* Naive Persistence
* Seasonal Persistence
* Random Forest
* Extra Trees
* Gradient Boosting
* HistGradientBoosting
* LightGBM
* XGBoost
* CatBoost
* LSTM

Different models were evaluated using forecasting metrics such as:

* MAE
* RMSE
* R² Score

---

## 9. Model Selection

After comparing the models, **LightGBM** performed best for the main forecasting task.

The selected model achieved approximately:

The final model was selected based on its overall forecasting performance.

Other trained models were also kept for comparison and evaluation.

---

## 10. Why Feast Feature Store?

**Feast** was used as the Feature Store for the project.

The main reason for using Feast was to separate **feature generation** from **model prediction**.

It provides a proper way to:

* Store features
* Manage feature definitions
* Retrieve features for prediction
* Keep training and serving features consistent
* Support an ML production workflow

The project uses Feast for retrieving the latest features before making the 72-hour forecast.

### Feast Setup

The project uses:

```text
Offline Store → DuckDB
Online Store  → SQLite
Registry      → Local Registry
```

DuckDB was used for the offline feature data because the dataset could be processed efficiently without using a cloud service.

---

## 11. Why Feast Instead of Hopsworks?

Hopsworks was initially considered for the Feature Store.

However, the project faced practical issues with the Hopsworks setup and cloud dependency.

For this internship project, **Feast was selected because it could run locally and integrate more easily with the existing project**.

This made the system:

* Easier to develop
* Easier to test
* Less dependent on external cloud services
* Suitable for the internship environment

---

## 12. Model Training Pipeline

The training pipeline automatically:

1. Loads the prepared data
2. Performs feature engineering
3. Prepares training data
4. Trains the models
5. Evaluates the models
6. Selects/saves the trained models
7. Updates the required project files

The trained models are stored and used by the forecasting dashboard.

---

## 13. CI/CD Pipeline

GitHub Actions was used to automate important project tasks.

The project contains automated workflows for:

### Feature Pipeline

Runs regularly to:

* Collect/update data
* Generate new features
* Update the Feature Store data

### Training Pipeline

Runs regularly to:

* Train the forecasting models
* Evaluate the models
* Update the trained model

### CI Validation

The CI workflow checks whether the project setup and required components are working correctly.

### CI/CD Flow

```text
New Data
   ↓
Feature Pipeline
   ↓
Feature Store
   ↓
Training Pipeline
   ↓
New Model
   ↓
Dashboard
```

This reduces manual work and makes the project closer to a production ML system.

---

## 14. Dashboard

A **Streamlit dashboard** was developed to present the forecasting results in an easy-to-understand way.

The dashboard connects the complete ML pipeline:

```text
Latest Data
     ↓
Feature Engineering
     ↓
Feast Feature Store
     ↓
Trained Model
     ↓
72-Hour Forecast
     ↓
Dashboard
```

### Dashboard Shows

* Current AQI information
* Forecasted AQI
* Hour-by-hour predictions
* 3-day forecast
* Forecast trends
* Latest available information
* Model-generated results

The dashboard provides a simple interface for viewing the predicted air quality without running the ML code manually.

---

## 15. Technology Stack

### Programming

* Python

### Data & ML

* Pandas
* NumPy
* Scikit-learn
* LightGBM
* XGBoost
* CatBoost
* TensorFlow/Keras

### Feature Store

* Feast
* DuckDB
* SQLite

### Data Source

* Open-Meteo API

### Dashboard

* Streamlit

### Automation

* GitHub Actions
* CI/CD

### Version Control

* Git
* GitHub
* Git LFS

---

## 16. Project Structure

```text
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

## 17. Key Project Results

The project successfully provides:

* **3-day AQI forecasting**
* **72 hourly predictions**
* Automated feature generation
* Feature Store integration
* Machine Learning model training
* Model evaluation and selection
* Automated CI/CD pipelines
* Interactive forecasting dashboard

---

## 18. Project Link

Streamlit App Link:


---

## 19. Conclusion

Karachi Air Intelligence is an end-to-end Machine Learning project for **72-hour AQI forecasting in Karachi**.

The project combines:

**Data Collection → EDA → Feature Engineering → Feast → Machine Learning → Model Evaluation → CI/CD → Dashboard**

The system demonstrates how an ML forecasting model can be developed and connected to an automated pipeline and user-friendly dashboard.
