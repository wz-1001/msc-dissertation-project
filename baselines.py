"""
baselines.py

Trivial reference models - always guess the majority class, or always
guess the mean. Every real model gets checked against these; if it
can't beat them it isn't learning anything.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
)


def majority_class_baseline(y_train, y_test):
    majority_class = pd.Series(y_train).value_counts().idxmax()

    train_preds = np.full(shape=len(y_train), fill_value=majority_class)
    test_preds = np.full(shape=len(y_test), fill_value=majority_class)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    # weighted average - accounts for class imbalance instead of treating every class equally
    test_precision = precision_score(y_test, test_preds, average="weighted", zero_division=0)
    test_recall = recall_score(y_test, test_preds, average="weighted", zero_division=0)
    test_f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)

    return {
        "Model": "Majority-Class Reference",
        "Family": "Reference",
        "Accuracy": round(test_acc, 4),
        "Train Accuracy": round(train_acc, 4),
        "Precision": round(test_precision, 4),
        "Recall": round(test_recall, 4),
        "F1 Score": round(test_f1, 4),
        "Emissions (gCO2)": 0,
    }


def mean_regression_baseline(y_train, y_test):
    mean_val = float(np.mean(y_train))
    train_preds = np.full(shape=len(y_train), fill_value=mean_val)
    test_preds = np.full(shape=len(y_test), fill_value=mean_val)

    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_rmse = float(mean_squared_error(y_test, test_preds) ** 0.5)  # sklearn only gives mse - sqrt it here for rmse

    return {
        "Model": "Mean-Value Reference",
        "Family": "Reference",
        "R2": round(test_r2, 4),
        "Train R2": round(train_r2, 4),
        "MAE": round(test_mae, 4),
        "RMSE": round(test_rmse, 4),
        "Emissions (gCO2)": 0,
    }