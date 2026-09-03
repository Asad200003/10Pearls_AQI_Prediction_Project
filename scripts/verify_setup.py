#!/usr/bin/env python3
"""
scripts/verify_setup.py
=========================
CI validation script (.github/workflows/ci_validation.yml). Runs on every push/PR.

This is a REAL smoke test, not a fake CI step: it actually exercises
write_offline_source -> generate_feature_views_file -> generate_feature_service_file
-> apply_feast_definitions -> materialize_feast_features -> get_online_features_for_
prediction against a throwaway synthetic dataset and a SCRATCH Feast repo directory
(never the real feature_store/ - this must never touch or depend on real project
state, so it can run standalone on a fresh checkout with no prior setup).

What it checks:
    1. Every shared module imports cleanly (feature_engineering, feast_utils, data_pipeline)
    2. dashboard/app.py and scripts/*.py parse without syntax errors
    3. The training notebook is valid (nbformat) and every code cell parses
    4. Feast's Entity/FeatureView/FeatureService objects are actually valid and
       accepted by `apply` (generated from a synthetic-but-realistic feature schema)
    5. Feast materialize actually populates the online store
    6. Online feature retrieval actually works end-to-end
    7. verify_feast_setup() correctly reports "ok" for a properly set-up store

Exit code 0 = all checks passed, non-zero = at least one failed (printed clearly).
"""

import os
import sys
import ast
import shutil
import tempfile
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FAILURES = []


def check(name):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            print(f"\n--- {name} ---")
            try:
                fn(*args, **kwargs)
                print(f"PASS: {name}")
                return True
            except Exception as e:
                print(f"FAIL: {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
                FAILURES.append(name)
                return False
        return wrapper
    return decorator


@check("Import shared modules")
def check_imports():
    import feature_engineering  # noqa: F401
    import feast_utils  # noqa: F401
    import data_pipeline  # noqa: F401
    print("  feature_engineering, feast_utils, data_pipeline import cleanly")


@check("Syntax-check all project Python files")
def check_syntax():
    targets = [
        "dashboard/app.py",
        "scripts/feature_pipeline.py",
        "scripts/train_pipeline.py",
        "feature_store/entities.py",
    ]
    for rel in targets:
        path = os.path.join(PROJECT_ROOT, rel)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            ast.parse(f.read(), filename=rel)
        print(f"  {rel}: OK")


@check("Validate the training notebook")
def check_notebook():
    import nbformat
    nb_path = os.path.join(PROJECT_ROOT, "notebooks", "Karachi_AQI_Forecasting.ipynb")
    nb = nbformat.read(nb_path, as_version=4)
    nbformat.validate(nb)
    errors = 0
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            try:
                ast.parse(c.source)
            except SyntaxError as e:
                errors += 1
                print(f"  SYNTAX ERROR in cell {i}: {e}")
    if errors:
        raise RuntimeError(f"{errors} notebook cell(s) failed to parse")
    print(f"  {len(nb.cells)} cells, all valid")


@check("Feast entity/FeatureView/FeatureService are valid and apply() succeeds")
def check_feast_apply(scratch_repo):
    import numpy as np
    import pandas as pd
    import feast_utils as fu
    from feature_engineering import (get_feature_columns, create_features, create_targets,
                                      TARGET_HORIZONS, WEATHER_HOURLY_VARS, AIR_QUALITY_HOURLY_VARS)

    np.random.seed(0)
    n = 24 * 10  # 10 days - just enough for a full lag/rolling feature set
    ts = pd.date_range("2025-01-01", periods=n, freq="h")
    df = pd.DataFrame({"timestamp": ts})
    for c in WEATHER_HOURLY_VARS:
        df[c] = np.random.rand(n) * 20 + 10
    for c in [v for v in AIR_QUALITY_HOURLY_VARS if v != "us_aqi"]:
        df[c] = np.random.rand(n) * 50 + 5
    df["us_aqi"] = np.clip(np.random.rand(n) * 150 + 20, 0, 500)

    feat_df = create_features(df)
    targ_df = create_targets(feat_df)
    model_df = targ_df.dropna(subset=[f"AQI_t+{h}" for h in TARGET_HORIZONS]).reset_index(drop=True)
    feature_cols = get_feature_columns(model_df)
    print(f"  synthetic schema: {len(feature_cols)} feature columns")

    fu.write_offline_source(model_df, feature_cols, parquet_path=os.path.join(scratch_repo, "data", "karachi_features.parquet"))
    fu.generate_feature_views_file(feature_cols, repo_path=scratch_repo)
    fu.generate_feature_service_file(repo_path=scratch_repo)
    fu.apply_feast_definitions(repo_path=scratch_repo)
    print("  apply() succeeded - Entity/FeatureView/FeatureService are valid")

    global _model_df_for_materialize
    _model_df_for_materialize = model_df


@check("Feast materialize populates the online store")
def check_feast_materialize(scratch_repo):
    import feast_utils as fu
    model_df = _model_df_for_materialize
    fu.materialize_feast_features(model_df["timestamp"].min(), model_df["timestamp"].max(), repo_path=scratch_repo)
    print("  materialize() succeeded")


@check("Online feature retrieval works end-to-end")
def check_online_retrieval(scratch_repo):
    import feast_utils as fu
    feature_cols = fu.get_registered_feature_columns(repo_path=scratch_repo)
    vec = fu.get_online_features_for_prediction(feature_cols, repo_path=scratch_repo)
    assert vec.shape[0] == 1 and vec.shape[1] == len(feature_cols), f"unexpected shape {vec.shape}"
    assert list(vec.columns) == feature_cols, "feature order mismatch"
    print(f"  online vector shape {vec.shape}, correctly ordered")


@check("verify_feast_setup() reports a healthy store")
def check_verify_helper(scratch_repo):
    import feast_utils as fu
    result = fu.verify_feast_setup(repo_path=scratch_repo)
    if not result["ok"]:
        raise RuntimeError(f"verify_feast_setup() reported not-ok: {result['message']}")
    print(f"  {result['message']}")


def main():
    print("=" * 60)
    print("CI VALIDATION - Karachi AQI Forecasting")
    print("=" * 60)

    check_imports()
    check_syntax()
    check_notebook()

    scratch_repo = tempfile.mkdtemp(prefix="feast_ci_scratch_")
    os.makedirs(os.path.join(scratch_repo, "data"), exist_ok=True)
    try:
        # Copy the static feature_store.yaml + entities.py into the scratch repo so
        # apply() has a valid config to work against, without touching the real
        # feature_store/ directory or its data/ contents.
        real_repo = os.path.join(PROJECT_ROOT, "feature_store")
        for fname in ["feature_store.yaml", "entities.py"]:
            shutil.copy(os.path.join(real_repo, fname), os.path.join(scratch_repo, fname))

        ok1 = check_feast_apply(scratch_repo)
        if ok1:
            ok2 = check_feast_materialize(scratch_repo)
            if ok2:
                check_online_retrieval(scratch_repo)
                check_verify_helper(scratch_repo)
    finally:
        shutil.rmtree(scratch_repo, ignore_errors=True)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"CI VALIDATION FAILED: {len(FAILURES)} check(s) failed: {FAILURES}")
        print("=" * 60)
        return 1
    else:
        print("CI VALIDATION PASSED: all checks succeeded.")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
