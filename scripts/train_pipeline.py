#!/usr/bin/env python3
"""
scripts/train_pipeline.py
============================
Standalone script for the DAILY GitHub Actions training pipeline
(.github/workflows/training_pipeline.yml).

WHY THIS SCRIPT EXECUTES THE NOTEBOOK INSTEAD OF RE-IMPLEMENTING IT:

The training pipeline in this project is genuinely large - data collection, cleaning,
feature engineering, Feast historical retrieval, chronological split, ~9 model families
(persistence baselines, Ridge, Random Forest, Extra Trees, Gradient Boosting /
HistGradientBoosting, LightGBM, XGBoost, CatBoost, LSTM), Optuna hyperparameter tuning,
a validation-weighted ensemble, full evaluation, error analysis, and model saving - all
already implemented and tested in notebooks/Karachi_AQI_Forecasting.ipynb.

Hand-porting that into a second, parallel .py implementation would be exactly the kind
of "second implementation of something that already exists" the internship
requirements explicitly forbid, and would risk silently drifting from the notebook's
actual (evaluated, working) behavior over time. Executing the real notebook
non-interactively with papermill is the only way to guarantee IDENTICAL behavior
between what a human runs in Jupyter and what CI runs automatically - it IS the
existing training pipeline, not a reimplementation of it.

Flow (matches the internship's required diagram exactly - this is literally what
notebook cells 1-96 already do):

    Feast historical features (fu.load_training_features_from_feast, already in the notebook)
        -> existing chronological train/validation/test split
        -> existing model training (all families)
        -> existing model evaluation
        -> existing model-selection logic (lowest validation MAE)
        -> existing model saving (models/best_model.pkl or ensemble files + model_metadata.json)

This script's own job is only: run the notebook end-to-end, verify the expected output
artifacts exist afterward, and give GitHub Actions a clean exit code + readable log.

Exit codes:
    0  - notebook executed successfully and expected model artifacts were produced
    1  - notebook execution failed (see the executed notebook / papermill traceback)
    2  - notebook "succeeded" but expected output artifacts are missing (should not
         normally happen; treated as a failure so CI doesn't silently report green)
"""

import os
import sys
import json
import traceback
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_PATH = os.path.join(PROJECT_ROOT, "notebooks", "Karachi_AQI_Forecasting.ipynb")
EXECUTED_NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")
EXECUTED_NOTEBOOK_PATH = os.path.join(EXECUTED_NOTEBOOK_DIR, "Karachi_AQI_Forecasting_executed.ipynb")
LOG_PATH = os.path.join(PROJECT_ROOT, "outputs", "metrics", "last_train_pipeline_run.json")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EXPECTED_ARTIFACTS = [
    os.path.join(MODELS_DIR, "model_config.json"),
    os.path.join(MODELS_DIR, "model_metadata.json"),
    os.path.join(MODELS_DIR, "feature_columns.json"),
    os.path.join(MODELS_DIR, "scaler.pkl"),
]


def _write_run_log(status, detail):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    payload = {
        "status": status,
        "detail": detail,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(LOG_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _artifact_check():
    """Confirms the notebook actually produced a usable model, per the notebook's own
    Section 24 'Save Model and Outputs' logic (models/best_model.pkl for a single
    winning model, OR models/ensemble_weights.json + per-member files for the
    validation-weighted ensemble - the notebook decides which at runtime based on
    which scored best on validation, exactly as before; this check just confirms
    *some* valid outcome of that existing logic landed on disk)."""
    missing = [p for p in EXPECTED_ARTIFACTS if not os.path.exists(p)]
    if missing:
        return False, missing

    config = json.load(open(os.path.join(MODELS_DIR, "model_config.json")))
    if config.get("final_model_name") == "Weighted Ensemble":
        if not os.path.exists(os.path.join(MODELS_DIR, "ensemble_weights.json")):
            return False, [os.path.join(MODELS_DIR, "ensemble_weights.json")]
    else:
        if not os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl")):
            return False, [os.path.join(MODELS_DIR, "best_model.pkl")]

    return True, []


def main():
    import papermill as pm

    print("=" * 60)
    print("DAILY TRAINING PIPELINE")
    print("=" * 60)
    print(f"Run started (UTC): {datetime.now(timezone.utc).isoformat()}")
    print(f"Executing notebook: {NOTEBOOK_PATH}")

    os.makedirs(EXECUTED_NOTEBOOK_DIR, exist_ok=True)

    # Execute the EXISTING notebook exactly as a human running "Restart & Run All"
    # would - no parameters are overridden here (FAST_MODE, START_DATE/END_DATE,
    # model architecture, evaluation methodology, and model-selection strategy all
    # stay exactly what the notebook's own config cell already says).
    try:
        pm.execute_notebook(
            input_path=NOTEBOOK_PATH,
            output_path=EXECUTED_NOTEBOOK_PATH,
            cwd=os.path.join(PROJECT_ROOT, "notebooks"),
            progress_bar=False,
            log_output=True,
        )
    except Exception:
        print("\n!!! NOTEBOOK EXECUTION FAILED !!!")
        traceback.print_exc()
        _write_run_log("failed", f"papermill execution error; see {EXECUTED_NOTEBOOK_PATH} for the "
                                  f"partially-executed notebook with the failing cell's traceback.")
        return 1

    print(f"\nNotebook executed successfully -> {EXECUTED_NOTEBOOK_PATH}")

    ok, missing = _artifact_check()
    if not ok:
        msg = f"Notebook ran without raising an error, but expected artifacts are missing: {missing}"
        print(f"\n!!! {msg} !!!")
        _write_run_log("failed_missing_artifacts", msg)
        return 2

    metadata = json.load(open(os.path.join(MODELS_DIR, "model_metadata.json")))
    detail = {
        "final_model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "n_features": metadata.get("n_features"),
        "executed_notebook": EXECUTED_NOTEBOOK_PATH,
    }
    _write_run_log("success", detail)

    print("\n" + "=" * 60)
    print(f"TRAINING PIPELINE COMPLETE - production model: {metadata.get('model_name')}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n!!! TRAINING PIPELINE FAILED (unexpected error) !!!")
        traceback.print_exc()
        try:
            _write_run_log("failed", traceback.format_exc())
        except Exception:
            pass
        sys.exit(1)
