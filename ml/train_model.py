"""
Trains a TensorFlow regression model to predict construction project
risk: cost overrun percentage and schedule delay (days), from project
features (size, suppliers, subcontractors, change orders, site
conditions, equipment utilization, etc).

Run:
    python ml/train_model.py
Outputs:
    ml/model/risk_model.keras
    ml/model/preprocessor.joblib
    ml/model/metrics.json   <- read by CI to gate regressions
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "size_sqm",
    "planned_budget_aed",
    "planned_duration_days",
    "num_suppliers",
    "supplier_reliability_score",
    "num_subcontractors",
    "design_change_orders",
    "weather_risk_days",
    "site_congestion_score",
    "equipment_utilization_pct",
    "permit_delay_days",
    "labor_turnover_pct",
]
CATEGORICAL_FEATURES = ["project_type"]
TARGETS = ["cost_overrun_pct", "delay_days"]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "construction_projects.csv")
MODEL_DIR = os.path.join(HERE, "model")


def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(2),  # [cost_overrun_pct, delay_days]
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae", metrics=["mae"])
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGETS].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    model = build_model(X_train_t.shape[1])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    model.fit(
        X_train_t,
        y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=64,
        callbacks=[early_stop],
        verbose=2,
    )

    preds = model.predict(X_test_t, verbose=0)
    mae_cost_overrun = float(np.mean(np.abs(preds[:, 0] - y_test[:, 0])))
    mae_delay_days = float(np.mean(np.abs(preds[:, 1] - y_test[:, 1])))

    print(f"Test MAE - cost overrun: {mae_cost_overrun:.2f} pct points")
    print(f"Test MAE - delay: {mae_delay_days:.2f} days")

    model.save(os.path.join(MODEL_DIR, "risk_model.keras"))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))
    joblib.dump(
        {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES, "targets": TARGETS},
        os.path.join(MODEL_DIR, "feature_spec.joblib"),
    )

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(
            {"test_mae_cost_overrun_pct": mae_cost_overrun, "test_mae_delay_days": mae_delay_days},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
