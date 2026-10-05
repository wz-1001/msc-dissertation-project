"""
models.py

Models for the benchmarking pipeline: the three complexity tiers
(Baseline, Ensemble, Deep Learning) and the custom Keras wrappers that
give tabular deep learning models a Scikit-Learn-style fit/predict API,
so benchmark.py can treat every model - classical or deep - identically.

Tier 3 models accept an optional validation set (X_val, y_val) used for
early stopping, in place of Keras' own internal validation_split. For
regression, the target is internally standard-scaled before training and
inverse-transformed on prediction, since an unscaled target in the
hundreds of thousands (e.g. price data) makes gradient-based training
much harder than it needs to be for no benefit. Dropout and L2 weight
regularisation were added to all three architectures - early stopping
alone wasn't giving the networks anything actively fighting overfitting
between epochs, just a point to stop at.

Classification models that support it now use class_weight='balanced',
so a skewed class distribution doesn't just get scikit-learn's default
majority-favouring behaviour. REGULARISATION_C documents the strength
used for the linear models rather than leaving it as an unexamined
default.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

SVM_SCALE_THRESHOLD = 20000  # past this many rows, rbf-kernel SVM/SVR gets impractically slow - switch to the linear variant
REGULARISATION_C = 1.0  # scikit-learn's own default - named here so it's a documented choice, not an implicit one
DL_DROPOUT_RATE = 0.2
DL_L2_REG = 1e-4


class TabularMLP:
    supports_validation_set = True

    def __init__(self, epochs=100, batch_size=32, task_type='classification'):
        self.epochs = epochs
        self.batch_size = batch_size
        self.task_type = task_type
        self.model = None
        self.num_classes = None
        self.y_scaler = None

    def fit(self, X, y, X_val=None, y_val=None):
        tf.keras.backend.clear_session()

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        l2 = tf.keras.regularizers.l2(DL_L2_REG)

        if self.task_type == 'regression':
            self.y_scaler = StandardScaler()
            y_scaled = self.y_scaler.fit_transform(np.asarray(y).reshape(-1, 1)).flatten()

            self.model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(X.shape[1],)),
                tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2),
                tf.keras.layers.Dropout(DL_DROPOUT_RATE),
                tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2),
                tf.keras.layers.Dense(1)
            ])
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            fit_kwargs = {"epochs": self.epochs, "batch_size": self.batch_size,
                          "callbacks": [early_stopping], "verbose": 0}
            if X_val is not None and y_val is not None:
                y_val_scaled = self.y_scaler.transform(np.asarray(y_val).reshape(-1, 1)).flatten()
                fit_kwargs["validation_data"] = (X_val, y_val_scaled)
            else:
                fit_kwargs["validation_split"] = 0.2

            self.model.fit(X, y_scaled, **fit_kwargs)
            return self

        self.num_classes = len(np.unique(y))
        is_binary = self.num_classes == 2

        output_units = 1 if is_binary else self.num_classes
        output_activation = 'sigmoid' if is_binary else 'softmax'
        loss = 'binary_crossentropy' if is_binary else 'sparse_categorical_crossentropy'

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2),
            tf.keras.layers.Dropout(DL_DROPOUT_RATE),
            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2),
            tf.keras.layers.Dense(output_units, activation=output_activation)
        ])
        self.model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

        fit_kwargs = {"epochs": self.epochs, "batch_size": self.batch_size,
                      "callbacks": [early_stopping], "verbose": 0}
        if X_val is not None and y_val is not None:
            fit_kwargs["validation_data"] = (X_val, y_val)
        else:
            fit_kwargs["validation_split"] = 0.2

        self.model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X):
        preds = self.model.predict(X, verbose=0)
        if self.task_type == 'regression':
            preds = preds.flatten()
            if self.y_scaler is not None:
                preds = self.y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            return preds
        if self.num_classes == 2:
            return (preds > 0.5).astype(int).flatten()
        return np.argmax(preds, axis=1)


class TabularCNN:
    supports_validation_set = True

    def __init__(self, epochs=100, batch_size=32, task_type='classification'):
        self.epochs = epochs
        self.batch_size = batch_size
        self.task_type = task_type
        self.model = None
        self.num_classes = None
        self.y_scaler = None

    def fit(self, X, y, X_val=None, y_val=None):
        tf.keras.backend.clear_session()
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        l2 = tf.keras.regularizers.l2(DL_L2_REG)

        if self.task_type == 'regression':
            self.y_scaler = StandardScaler()
            y_scaled = self.y_scaler.fit_transform(np.asarray(y).reshape(-1, 1)).flatten()

            self.model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(X.shape[1], 1)),
                tf.keras.layers.Conv1D(32, 2, activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2),
                tf.keras.layers.Dropout(DL_DROPOUT_RATE),
                tf.keras.layers.Dense(1)
            ])
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            fit_kwargs = {"epochs": self.epochs, "batch_size": self.batch_size,
                          "callbacks": [early_stopping], "verbose": 0}
            if X_val is not None and y_val is not None:
                X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
                y_val_scaled = self.y_scaler.transform(np.asarray(y_val).reshape(-1, 1)).flatten()
                fit_kwargs["validation_data"] = (X_val_reshaped, y_val_scaled)
            else:
                fit_kwargs["validation_split"] = 0.2

            self.model.fit(X_reshaped, y_scaled, **fit_kwargs)
            return self

        self.num_classes = len(np.unique(y))
        is_binary = self.num_classes == 2

        output_units = 1 if is_binary else self.num_classes
        output_activation = 'sigmoid' if is_binary else 'softmax'
        loss = 'binary_crossentropy' if is_binary else 'sparse_categorical_crossentropy'

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X.shape[1], 1)),
            tf.keras.layers.Conv1D(32, 2, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2),
            tf.keras.layers.Dropout(DL_DROPOUT_RATE),
            tf.keras.layers.Dense(output_units, activation=output_activation)
        ])
        self.model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

        fit_kwargs = {"epochs": self.epochs, "batch_size": self.batch_size,
                      "callbacks": [early_stopping], "verbose": 0}
        if X_val is not None and y_val is not None:
            X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
            fit_kwargs["validation_data"] = (X_val_reshaped, y_val)
        else:
            fit_kwargs["validation_split"] = 0.2

        self.model.fit(X_reshaped, y, **fit_kwargs)
        return self

    def predict(self, X):
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        preds = self.model.predict(X_reshaped, verbose=0)
        if self.task_type == 'regression':
            preds = preds.flatten()
            if self.y_scaler is not None:
                preds = self.y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            return preds
        if self.num_classes == 2:
            return (preds > 0.5).astype(int).flatten()
        return np.argmax(preds, axis=1)


class TabularLSTM:
    supports_validation_set = True

    def __init__(self, epochs=100, batch_size=32, task_type='classification'):
        self.epochs = epochs
        self.batch_size = batch_size
        self.task_type = task_type
        self.model = None
        self.num_classes = None
        self.y_scaler = None

    def fit(self, X, y, X_val=None, y_val=None):
        tf.keras.backend.clear_session()
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        l2 = tf.keras.regularizers.l2(DL_L2_REG)

        if self.task_type == 'regression':
            self.y_scaler = StandardScaler()
            y_scaled = self.y_scaler.fit_transform(np.asarray(y).reshape(-1, 1)).flatten()

            self.model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(X.shape[1], 1)),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2),
                tf.keras.layers.Dropout(DL_DROPOUT_RATE),
                tf.keras.layers.Dense(1)
            ])
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            fit_kwargs = {"epochs": self.epochs, "batch_size": self.batch_size,
                          "callbacks": [early_stopping], "verbose": 0}
            if X_val is not None and y_val is not None:
                X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
                y_val_scaled = self.y_scaler.transform(np.asarray(y_val).reshape(-1, 1)).flatten()
                fit_kwargs["validation_data"] = (X_val_reshaped, y_val_scaled)
            else:
                fit_kwargs["validation_split"] = 0.2

            self.model.fit(X_reshaped, y_scaled, **fit_kwargs)
            return self

        self.num_classes = len(np.unique(y))
        is_binary = self.num_classes == 2

        output_units = 1 if is_binary else self.num_classes
        output_activation = 'sigmoid' if is_binary else 'softmax'
        loss = 'binary_crossentropy' if is_binary else 'sparse_categorical_crossentropy'

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X.shape[1], 1)),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2),
            tf.keras.layers.Dropout(DL_DROPOUT_RATE),
            tf.keras.layers.Dense(output_units, activation=output_activation)
        ])
        self.model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

        fit_kwargs = {"epochs": self.epochs, "batch_size": self.batch_size,
                      "callbacks": [early_stopping], "verbose": 0}
        if X_val is not None and y_val is not None:
            X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
            fit_kwargs["validation_data"] = (X_val_reshaped, y_val)
        else:
            fit_kwargs["validation_split"] = 0.2

        self.model.fit(X_reshaped, y, **fit_kwargs)
        return self

    def predict(self, X):
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        preds = self.model.predict(X_reshaped, verbose=0)
        if self.task_type == 'regression':
            preds = preds.flatten()
            if self.y_scaler is not None:
                preds = self.y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            return preds
        if self.num_classes == 2:
            return (preds > 0.5).astype(int).flatten()
        return np.argmax(preds, axis=1)


def build_models(num_classes: int, n_samples: int = None, task_type: str = "classification", is_time_series: bool = False):
    baseline_models = {}
    ensemble_models = {
        "Random Forest": (
            RandomForestRegressor(n_estimators=50, max_depth=15)
            if task_type == "regression"
            else RandomForestClassifier(n_estimators=50, max_depth=15, class_weight='balanced')
        ),
        "XGBoost": (
            XGBRegressor(eval_metric='rmse', max_depth=5, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8)
            if task_type == "regression"
            else XGBClassifier(eval_metric='logloss' if num_classes == 2 else 'mlogloss',
                                max_depth=5, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8)
        ),
    }

    dl_models = {}

    if task_type == "regression":
        baseline_models["Decision Tree"] = DecisionTreeRegressor(max_depth=8, min_samples_leaf=5)
        baseline_models["Linear Regression"] = LinearRegression()
        if n_samples is not None and n_samples > SVM_SCALE_THRESHOLD:
            baseline_models["SVM"] = LinearSVR(C=REGULARISATION_C, max_iter=5000)
        else:
            baseline_models["SVM"] = SVR(C=REGULARISATION_C, kernel='rbf')
    else:
        baseline_models["Decision Tree"] = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight='balanced')
        baseline_models["Naive Bayes"] = GaussianNB()  # no class_weight parameter available for this model
        baseline_models["Logistic Regression"] = LogisticRegression(C=REGULARISATION_C, max_iter=1000, class_weight='balanced')
        if n_samples is not None and n_samples > SVM_SCALE_THRESHOLD:
            baseline_models["SVM"] = LinearSVC(C=REGULARISATION_C, max_iter=2000, class_weight='balanced')
        else:
            baseline_models["SVM"] = SVC(C=REGULARISATION_C, kernel='rbf', class_weight='balanced')

    if n_samples is None or n_samples >= 500:
        dl_models["MLP"] = TabularMLP(epochs=100, task_type=task_type)

        # cnn/lstm assume order between adjacent features means something - only true for the lag columns a real time series produces
        if is_time_series:
            dl_models["CNN"] = TabularCNN(epochs=100, task_type=task_type)
            dl_models["LSTM"] = TabularLSTM(epochs=100, task_type=task_type)

    return {
        "Baseline": baseline_models,
        "Ensemble": ensemble_models,
        "Deep Learning": dl_models,
    }