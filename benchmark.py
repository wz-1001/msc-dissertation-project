"""
benchmark.py

turns an uploaded dataframe into a finished results table. handles
target-column prep (including turning an ons-style time series into
lagged features), hands leakage scoring off to column_analysis.py,
trains every model from models.py, and lets trackers.py measure
emissions per model. run_experiment() is the one function the api and
dashboard actually call.

default split is a genuine 60/20/20 train/val/test. test stays fully
held out for the headline accuracy/r2 and for the "at or below
baseline" check, since that's the model's real final performance, not
a training diagnostic. the overfitting note compares train vs
validation instead, since that's the usual pairing for a
during-training diagnostic - test stays untouched for final reporting.
validation also goes to any model with supports_validation_set (the
tier 3 neural nets, for early stopping).

every model also reports overfit gap (train minus validation) as data
only - this file never uses it to change rankings. what app.py does
with that number is its business.

for regression, if the training target is skewed enough (measured, not
guessed), models fit against log1p(target) and predictions get
inverse-transformed before any metric is touched.
"""

import os
import time
import tempfile
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

from models import build_models
from column_analysis import compute_column_associations
from baselines import majority_class_baseline, mean_regression_baseline
from trackers import get_tracker, ALLOWED_LIBRARIES

LOG_TRANSFORM_SKEW_THRESHOLD = 1.0  # above this the target counts as skewed enough for log1p


def _resolve_output_dir() -> str:
    preferred = "./output"
    try:
        os.makedirs(preferred, exist_ok=True)
        probe = os.path.join(preferred, ".write_test")
        with open(probe, "w") as f:
            f.write("")
        os.remove(probe)
        return preferred
    except OSError as e:
        fallback = os.path.join(tempfile.gettempdir(), "green_ai_output")
        os.makedirs(fallback, exist_ok=True)
        print(f"[benchmark.py] '{preferred}' isn't writable here ({e}) - using '{fallback}' instead.")
        return fallback


OUTPUT_DIR = _resolve_output_dir()


def _looks_like_time_index(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return False
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() > 0.8)


def looks_like_time_series(df: pd.DataFrame) -> bool:
    """lets the api check a raw upload's shape before a target column is even picked."""
    return bool(df.shape[1] == 2 and _looks_like_time_index(df.iloc[:, 0]))


def should_skip_association_check(df: pd.DataFrame) -> bool:
    """lag_1/2/3 are supposed to correlate with the target - that's the point of them,
    so running the leakage check on them would just flag good signal as a problem."""
    return "lag_1" in df.columns


def _resolve_task_type(series: pd.Series, requested: str) -> str:
    if requested in ("classification", "regression"):
        return requested
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
        return "regression"
    return "classification"


def prepare_target(df: pd.DataFrame, target_col: str, dataset_name: str, task_type: str = "auto"):
    df = df.dropna(subset=[target_col])

    is_time_series = False

    if df.shape[1] == 2 and _looks_like_time_index(df.iloc[:, 0]):
        print(f"Detected time-series structure in {dataset_name}. Converting to lagged features...")
        is_time_series = True
        val_col = df.columns[1]

        window_size = 3
        ts_df = pd.DataFrame()
        for i in range(window_size, 0, -1):
            ts_df[f'lag_{i}'] = df[val_col].shift(i)
        ts_df['target'] = df[val_col]

        df = ts_df.dropna()
        target_col = 'target'

    resolved_task = _resolve_task_type(df[target_col], task_type)

    if resolved_task == "regression":
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            return None, None, {
                "Model": "Data Error", "Family": "Error", "R2": 0, "MAE": 0, "RMSE": 0,
                "Emissions (gCO2)": 0,
                "Error": "Regression was requested but the target column is not numeric."
            }, resolved_task, is_time_series
        df[target_col] = df[target_col].astype(float)

    else:
        if pd.api.types.is_numeric_dtype(df[target_col]) and len(df[target_col].unique()) > 10:
            median_val = df[target_col].median()
            new_target = (df[target_col] > median_val).astype(int)
            if len(new_target.unique()) == 1:
                new_target = (df[target_col] >= df[target_col].mean()).astype(int)
            df[target_col] = new_target
        else:
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col])

        if len(df[target_col].unique()) <= 1:
            return None, None, {
                "Model": "Data Error", "Family": "Error", "Accuracy": 0, "F1 Score": 0,
                "Emissions (gCO2)": 0,
                "Error": "Target column only has 1 unique class. ML requires at least 2."
            }, resolved_task, is_time_series

    return df, target_col, None, resolved_task, is_time_series


def _prepare_dataset(df: pd.DataFrame, target_col: str, dataset_name: str,
                      task_type: str = "auto", excluded_columns: list = None):

    df, target_col, error, resolved_task, is_time_series = prepare_target(df, target_col, dataset_name, task_type)
    if error is not None:
        return None, None, error, [], resolved_task, is_time_series

    if should_skip_association_check(df):
        print(f"[{dataset_name}] Time series detected. Bypassing leakage analysis.")
        return df, target_col, None, [], resolved_task, is_time_series

    associations = compute_column_associations(df, target_col, task_type=resolved_task)

    if excluded_columns is not None:
        drop_cols = [c for c in excluded_columns if c in df.columns and c != target_col]
        chosen_by_user = True
    else:
        drop_cols = [a["column"] for a in associations if a["suggested_drop"]]
        chosen_by_user = False

    remaining = [c for c in df.columns if c != target_col and c not in drop_cols]

    if not remaining:
        if chosen_by_user:
            return None, None, {
                "Model": "Data Error", "Family": "Error", "Emissions (gCO2)": 0,
                "Error": "Every feature column was excluded - nothing left to train on. Keep at least one column and try again."
            }, associations, resolved_task, is_time_series
        else:
            drop_cols = []

    if drop_cols:
        print(
            f"[{dataset_name}] Dropping columns "
            f"({'user-selected' if chosen_by_user else 'auto'}): {drop_cols}"
        )
        df = df.drop(columns=drop_cols)

    for a in associations:
        a["dropped"] = a["column"] in drop_cols

    return df, target_col, None, associations, resolved_task, is_time_series


def _resample_to_size(df: pd.DataFrame, size: int, random_state: int = 42):
    n = len(df)
    if size <= n:
        return df.sample(n=size, random_state=random_state).reset_index(drop=True), False
    return df.sample(n=size, replace=True, random_state=random_state).reset_index(drop=True), True


def _benchmark_models(df: pd.DataFrame, target_col: str, dataset_name: str, emissions_library: str,
                       size_label=None, is_synthetic: bool = False,
                       tdp_watts: float = None, grid_intensity_g_per_kwh: float = None, pue: float = None,
                       compare_emissions: bool = False, task_type: str = "classification",
                       is_time_series: bool = False):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    for col in X.select_dtypes(include="object").columns:
        if X[col].nunique() > 50:
            X[col] = X[col].astype("category").cat.codes
    X = pd.get_dummies(X, drop_first=True)

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    def _tag(row):
        row["Task Type"] = task_type
        if size_label is not None:
            row["Sample Size"] = size_label
            row["Synthetic (Bootstrapped)"] = is_synthetic
        return row

    if task_type == "classification":
        class_counts = y.value_counts()
        if class_counts.min() < 2:
            return [_tag({"Model": "Data Imbalance", "Family": "Error", "Accuracy": 0, "F1 Score": 0, "Emissions (gCO2)": 0, "Error": f"Extreme imbalance! The minority class only has {class_counts.min()} row(s). ML needs more examples to train."})]

        try:
            X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42, stratify=y)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        except ValueError:
            X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        if len(np.unique(y_train)) < 2:
            return [_tag({"Model": "Data Split Error", "Family": "Error", "Accuracy": 0, "F1 Score": 0, "Emissions (gCO2)": 0, "Error": "After splitting data, the training set only had 1 class left."})]

        log_transform_target = False  # regression-only, never applies here

    else:  # regression
        if len(y) < 15:
            return [_tag({"Model": "Data Error", "Family": "Error", "R2": 0, "MAE": 0, "RMSE": 0, "Emissions (gCO2)": 0, "Error": "Not enough rows to train a regression model (need at least 15 for a train/validation/test split)."})]

        X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        try:
            target_skew = float(skew(y_train))
        except Exception:
            target_skew = 0.0
        log_transform_target = bool(y_train.min() >= 0 and target_skew > LOG_TRANSFORM_SKEW_THRESHOLD)
        if log_transform_target:
            print(
                f"[{dataset_name}] Training target is right-skewed (skew={target_skew:.2f}) - "
                f"fitting against log1p(target), reporting metrics on the inverse-transformed scale."
            )

    if emissions_library not in ALLOWED_LIBRARIES:
        return [_tag({
            "Model": "Config Error", "Family": "Error",
            "Emissions (gCO2)": 0,
            "Error": f"Unknown emissions library '{emissions_library}'. Choose one of: {', '.join(ALLOWED_LIBRARIES)}"
        })]

    y_train_fit = np.log1p(y_train) if log_transform_target else y_train

    num_classes = len(np.unique(y_train)) if task_type == "classification" else None
    size_note = f" | sample size: {size_label}{' (synthetic bootstrap)' if is_synthetic else ''}" if size_label is not None else ""

    print(
        f"[{dataset_name}] Train shape: {X_train.shape} | Test shape: {X_test.shape} "
        f"| Features: {X.shape[1]} | Task: {task_type}{size_note}"
    )

    MODELS = build_models(num_classes, n_samples=len(X_train), task_type=task_type, is_time_series=is_time_series)

    tracker_kwargs = {"output_dir": OUTPUT_DIR}
    if tdp_watts is not None:
        tracker_kwargs["tdp_watts"] = tdp_watts
    if grid_intensity_g_per_kwh is not None:
        tracker_kwargs["grid_intensity_g_per_kwh"] = grid_intensity_g_per_kwh
    if pue is not None:
        tracker_kwargs["pue"] = pue

    if task_type == "classification":
        results = [_tag(majority_class_baseline(y_train, y_test))]
        baseline_perf = results[0]["Accuracy"]
        majority_class = pd.Series(y_train).value_counts().idxmax()
        val_baseline_preds = np.full(shape=len(y_val), fill_value=majority_class)
        results[0]["Val Accuracy"] = round(accuracy_score(y_val, val_baseline_preds), 4)
        results[0]["Overfit Gap"] = round(results[0]["Train Accuracy"] - results[0]["Val Accuracy"], 4)
    else:
        results = [_tag(mean_regression_baseline(y_train, y_test))]
        baseline_perf = results[0]["R2"]
        mean_val = float(np.mean(y_train))
        val_baseline_preds = np.full(shape=len(y_val), fill_value=mean_val)
        results[0]["Val R2"] = round(r2_score(y_val, val_baseline_preds), 4)
        results[0]["Overfit Gap"] = round(results[0]["Train R2"] - results[0]["Val R2"], 4)

    results[0]["Training Time (s)"] = 0.0

    if compare_emissions:
        results[0]["Emissions CodeCarbon (gCO2)"] = 0
        results[0]["Emissions Manual (gCO2)"] = 0
        results[0]["Emissions Variance (%)"] = 0

    for family, models in MODELS.items():
        for model_name, model in models.items():
            row = {"Model": model_name, "Family": family}
            uses_val_set = getattr(model, "supports_validation_set", False)
            try:
                if compare_emissions:
                    primary_tracker = get_tracker("codecarbon", **tracker_kwargs)
                    secondary_tracker = get_tracker("manual", **tracker_kwargs)
                    primary_tracker.start()
                    secondary_tracker.start()

                    fit_start = time.perf_counter()
                    if uses_val_set:
                        model.fit(X_train, y_train_fit, X_val=X_val, y_val=y_val)
                    else:
                        model.fit(X_train, y_train_fit)
                    row["Training Time (s)"] = round(time.perf_counter() - fit_start, 4)

                    train_preds = model.predict(X_train)
                    test_preds = model.predict(X_test)

                    primary_kg = primary_tracker.stop()
                    secondary_kg = secondary_tracker.stop()

                    primary_g = round(primary_kg * 1000, 6)
                    secondary_g = round(secondary_kg * 1000, 6)

                    row["Emissions CodeCarbon (gCO2)"] = primary_g
                    row["Emissions Manual (gCO2)"] = secondary_g

                    if primary_g > 0:
                        variance_pct = ((secondary_g - primary_g) / primary_g) * 100
                    else:
                        variance_pct = 0.0

                    row["Emissions Variance (%)"] = round(variance_pct, 2)
                    row["Emissions (gCO2)"] = primary_g
                else:
                    tracker = get_tracker(emissions_library, **tracker_kwargs)
                    tracker.start()
                    
                    fit_start = time.perf_counter()
                    if uses_val_set:
                        model.fit(X_train, y_train_fit, X_val=X_val, y_val=y_val)
                    else:
                        model.fit(X_train, y_train_fit)
                    row["Training Time (s)"] = round(time.perf_counter() - fit_start, 4)

                    train_preds = model.predict(X_train)
                    test_preds = model.predict(X_test)

                    emissions_kg = tracker.stop()
                    row["Emissions (gCO2)"] = round(emissions_kg * 1000, 6)

                val_preds = model.predict(X_val)

                if log_transform_target:
                    train_preds = np.expm1(train_preds)
                    test_preds = np.expm1(test_preds)
                    if val_preds is not None:
                        val_preds = np.expm1(val_preds)

                if task_type == "classification":
                    train_acc = accuracy_score(y_train, train_preds)
                    test_acc = accuracy_score(y_test, test_preds)
                    val_acc = accuracy_score(y_val, val_preds)

                    row["Accuracy"] = round(test_acc, 4)
                    row["Train Accuracy"] = round(train_acc, 4)
                    row["Val Accuracy"] = round(val_acc, 4)
                    row["Overfit Gap"] = round(train_acc - val_acc, 4)
                    row["Precision"] = round(precision_score(y_test, test_preds, average="weighted", zero_division=0), 4)
                    row["Recall"] = round(recall_score(y_test, test_preds, average="weighted", zero_division=0), 4)
                    row["F1 Score"] = round(f1_score(y_test, test_preds, average="weighted"), 4)

                    if test_acc <= baseline_perf:
                        row["Note"] = "At or below majority-class reference - not learning useful signal"
                    elif train_acc - val_acc > 0.25:
                        row["Note"] = "Large train/validation gap - likely overfitting"

                    print(
                        f"  [{dataset_name}]{size_note} {model_name}: train_acc={train_acc:.4f} "
                        f"val_acc={val_acc:.4f} test_acc={test_acc:.4f} time_s={row['Training Time (s)']:.4f} "
                        f"emissions_g={row['Emissions (gCO2)']:.6f}"
                    )

                else:
                    test_rmse = float(mean_squared_error(y_test, test_preds) ** 0.5)
                    test_mae = mean_absolute_error(y_test, test_preds)
                    train_r2 = r2_score(y_train, train_preds)
                    test_r2 = r2_score(y_test, test_preds)
                    val_r2 = r2_score(y_val, val_preds)

                    row["R2"] = round(test_r2, 4)
                    row["Train R2"] = round(train_r2, 4)
                    row["Val R2"] = round(val_r2, 4)
                    row["Overfit Gap"] = round(train_r2 - val_r2, 4)
                    row["MAE"] = round(test_mae, 4)
                    row["RMSE"] = round(test_rmse, 4)
                    row["Log-Transformed Target"] = log_transform_target

                    if test_r2 <= baseline_perf:
                        row["Note"] = "At or below mean-value reference - not learning useful signal"
                    elif train_r2 - val_r2 > 0.25:
                        row["Note"] = "Large train/validation R2 gap - likely overfitting"

                    print(
                        f"  [{dataset_name}]{size_note} {model_name}: train_r2={train_r2:.4f} "
                        f"val_r2={val_r2:.4f} test_r2={test_r2:.4f} rmse={test_rmse:.4f} "
                        f"time_s={row['Training Time (s)']:.4f} emissions_g={row['Emissions (gCO2)']:.6f}"
                    )

            except Exception as e:
                row.setdefault("Training Time (s)", 0.0)
                row.setdefault("Overfit Gap", 0.0)
                if task_type == "classification":
                    row["Accuracy"] = 0
                    row["Train Accuracy"] = 0
                    row["Val Accuracy"] = 0
                    row["Precision"] = 0
                    row["Recall"] = 0
                    row["F1 Score"] = 0
                else:
                    row["R2"] = 0
                    row["Train R2"] = 0
                    row["Val R2"] = 0
                    row["MAE"] = 0
                    row["RMSE"] = 0
                row["Emissions (gCO2)"] = 0
                if compare_emissions:
                    row["Emissions CodeCarbon (gCO2)"] = 0
                    row["Emissions Manual (gCO2)"] = 0
                    row["Emissions Variance (%)"] = 0
                row["Error"] = f"{type(e).__name__}: {e}"

            results.append(_tag(row))

    return results


def run_experiment(df, target_col, dataset_name="Dataset", emissions_library="codecarbon",
                    sample_sizes=None,
                    tdp_watts: float = None, grid_intensity_g_per_kwh: float = None, pue: float = None,
                    compare_emissions: bool = False, task_type: str = "auto",
                    excluded_columns: list = None):
    print(f"Starting benchmark for {dataset_name}...")

    clean_df, resolved_target_col, error, column_associations, resolved_task, is_time_series = _prepare_dataset(
        df, target_col, dataset_name, task_type=task_type, excluded_columns=excluded_columns
    )
    if error is not None:
        return [error], column_associations, resolved_task

    if not sample_sizes:
        results = _benchmark_models(
            clean_df, resolved_target_col, dataset_name, emissions_library,
            tdp_watts=tdp_watts, grid_intensity_g_per_kwh=grid_intensity_g_per_kwh, pue=pue,
            compare_emissions=compare_emissions, task_type=resolved_task,
            is_time_series=is_time_series
        )
        return results, column_associations, resolved_task

    all_results = []
    for size in sample_sizes:
        sub_df, is_synthetic = _resample_to_size(clean_df, size)
        print(f"[{dataset_name}] --- Sweep: sample size {size} (synthetic bootstrap: {is_synthetic}) ---")
        rows = _benchmark_models(
            sub_df, resolved_target_col, dataset_name, emissions_library,
            size_label=size, is_synthetic=is_synthetic,
            tdp_watts=tdp_watts, grid_intensity_g_per_kwh=grid_intensity_g_per_kwh, pue=pue,
            compare_emissions=compare_emissions, task_type=resolved_task,
            is_time_series=is_time_series
        )
        all_results.extend(rows)

    return all_results, column_associations, resolved_task