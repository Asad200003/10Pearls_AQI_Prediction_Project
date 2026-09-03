#!/usr/bin/env python3
"""
scripts/feature_pipeline.py
=============================
Standalone script for the HOURLY GitHub Actions feature pipeline
(.github/workflows/feature_pipeline.yml).

Flow (matches the internship's required diagram exactly):

    Open-Meteo
        -> data_pipeline.fetch_recent_weather_and_air_quality()   (existing data collection)
        -> data_pipeline.merge_datasets()                          (existing merge logic)
        -> feature_engineering.create_features()                   (existing feature engineering)
        -> feast_utils.ingest_latest_observation()                 (existing Feast integration)
        -> Feast offline store (Parquet, via DuckDB offline store)
        -> Feast materialize-incremental
        -> Feast online store (SQLite)

Every step above calls a function that already exists in this project
(data_pipeline.py / feature_engineering.py / feast_utils.py) - nothing here
reimplements data collection, feature engineering, or Feast logic. This script is
only the orchestration + CI-friendly logging/exit-codes wrapper around them.

Exit codes:
    0  - ran successfully, a new feature row was ingested and materialized
    1  - unexpected error
    2  - Feast is not yet applied in this checkout (no registered FeatureView) - this
         is the expected state on a brand-new clone before the training pipeline has
         run at least once; not a bug, just "nothing to do yet"
"""

import os
import sys
import json
import traceback
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import feast_utils as fu  # noqa: E402
import data_pipeline as dp  # noqa: E402
from feature_engineering import create_features  # noqa: E402

LOG_PATH = os.path.join(PROJECT_ROOT, "feature_store", "data", "last_feature_pipeline_run.json")


def _write_run_log(status, detail):
    """Small JSON marker file, committed back to the repo by the GitHub Actions
    workflow alongside the Feast state - this is the easiest way to verify the hourly
    pipeline actually ran (see README 'How to verify the hourly feature pipeline ran')
    without digging through Actions logs."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    payload = {
        "status": status,
        "detail": detail,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(LOG_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def main():
    print("=" * 60)
    print("HOURLY FEATURE PIPELINE")
    print("=" * 60)
    print(f"Run started (UTC): {datetime.now(timezone.utc).isoformat()}")

    # --- 0. Make sure Feast is actually applied in this checkout. On a brand new
    # clone/runner this can be false if the training pipeline hasn't run yet (the
    # generated feature_store/feature_views.py + feature_service.py are what
    # `feast apply` needs, and they're produced by the training pipeline / notebook).
    # This is not duplicating Feast logic - apply_feast_definitions() IS the existing
    # feast_utils function, just re-run defensively/idempotently.
    try:
        fu.apply_feast_definitions()
    except fu.FeastConfigError as e:
        msg = (f"Feast is not yet set up in this checkout ({e}). This is expected if "
               f"the training pipeline hasn't run at least once yet - it's what "
               f"generates feature_store/feature_views.py and feature_service.py. "
               f"Nothing to ingest this run.")
        print(msg)
        _write_run_log("skipped_not_applied", msg)
        return 2

    # --- 1. Resolve the currently-registered feature schema from Feast itself (not
    # from a possibly-absent models/feature_columns.json) - existing feast_utils logic.
    try:
        feature_cols = fu.get_registered_feature_columns()
    except fu.FeastConfigError as e:
        msg = f"Could not read the registered FeatureView schema: {e}"
        print(msg)
        _write_run_log("failed", msg)
        return 2
    print(f"Registered feature columns: {len(feature_cols)}")

    # --- 1b. Bootstrap: if apply() just succeeded above but this online store has
    # never actually been materialized (e.g. the very first hourly run right after a
    # fresh training run in this checkout), materialize_incremental() below has
    # nothing to increment FROM - fill in the full historical range once, using the
    # same narrowly-scoped self-heal the dashboard uses, before proceeding.
    attempted, bootstrap_result = fu.bootstrap_materialize_if_needed()
    if attempted:
        print(f"Bootstrap materialize: {bootstrap_result['message']}")

    # --- 2. Existing data collection logic (data_pipeline.py) ---
    print("\nFetching recent weather + air quality from Open-Meteo...")
    weather_df, air_df = dp.fetch_recent_weather_and_air_quality(
        past_days=10, weather_forecast_days=1, air_forecast_days=1
    )
    print(f"  weather_df: {weather_df.shape}, air_df: {air_df.shape}")

    # --- 3. Existing feature engineering + existing Feast integration, via the single
    # shared ingestion function also used by dashboard/app.py (feast_utils.py) ---
    print("\nEngineering features and ingesting the latest observation into Feast...")
    observed, _future_weather, latest_row = fu.ingest_latest_observation(
        weather_df, air_df,
        create_features_fn=create_features,
        merge_datasets_fn=dp.merge_datasets,
        feature_cols=feature_cols,
    )

    latest_ts = latest_row["timestamp"].iloc[0]
    print(f"\nIngested feature row for timestamp: {latest_ts}")
    print(f"Observed history window used: {observed['timestamp'].min()} -> {observed['timestamp'].max()} "
          f"({len(observed)} hours)")

    detail = {
        "latest_feature_timestamp": str(latest_ts),
        "observed_window_start": str(observed["timestamp"].min()),
        "observed_window_end": str(observed["timestamp"].max()),
        "observed_rows": int(len(observed)),
        "n_features": len(feature_cols),
    }
    _write_run_log("success", detail)

    print("\n" + "=" * 60)
    print("FEATURE PIPELINE COMPLETE - Feast online store updated.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n!!! FEATURE PIPELINE FAILED !!!")
        traceback.print_exc()
        try:
            _write_run_log("failed", traceback.format_exc())
        except Exception:
            pass
        sys.exit(1)
