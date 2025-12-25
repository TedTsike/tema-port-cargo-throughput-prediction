"""
Model training module for Tema Port cargo throughput prediction.

This script:
- Trains multiple regression models
- Evaluates performance using MAE, RMSE, and MAPE
- Trains baseline and lag-augmented models
- Saves trained models for downstream evaluation and prediction
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate regression model performance.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    model_name : str
        Name of the model.
    """
    print(model_name)
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("MAPE (%):", np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    print("-" * 40)


# -------------------------------------------------------------------
# Model training functions
# -------------------------------------------------------------------

def train_baseline_models(X_train, X_test, y_train, y_test):
    """
    Train baseline machine learning models (no cargo lags).

    Returns
    -------
    dict
        Dictionary containing trained models and predictions.
    """
    models = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["Linear Regression"] = (lr, lr.predict(X_test))

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = (rf, rf.predict(X_test))

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    models["XGBoost"] = (xgb, xgb.predict(X_test))

    # Gradient Boosting
    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    gbr.fit(X_train, y_train)
    models["Gradient Boosting"] = (gbr, gbr.predict(X_test))

    return models


def train_lagged_models(X_train, X_test, y_train, y_test):
    """
    Train machine learning models using lag-augmented features.

    Returns
    -------
    dict
        Dictionary containing trained lagged models and predictions.
    """
    models = {}

    lrlag = LinearRegression()
    lrlag.fit(X_train, y_train)
    models["Linear Regression (Lagged)"] = (lrlag, lrlag.predict(X_test))

    rflag = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
    rflag.fit(X_train, y_train)
    models["Random Forest (Lagged)"] = (rflag, rflag.predict(X_test))

    xgblag = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    xgblag.fit(X_train, y_train)
    models["XGBoost (Lagged)"] = (xgblag, xgblag.predict(X_test))

    gbrlag = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    gbrlag.fit(X_train, y_train)
    models["Gradient Boosting (Lagged)"] = (gbrlag, gbrlag.predict(X_test))

    return models


# -------------------------------------------------------------------
# Prediction on unseen data
# -------------------------------------------------------------------

def predict_unseen_data(model, scaler, df):
    """
    Predict cargo throughput for new unseen operational and macroeconomic data.

    Parameters
    ----------
    model : trained model
        Trained regression model.
    scaler : StandardScaler
        Fitted scaler.
    df : pd.DataFrame
        Historical dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame containing prediction result.
    """
    new_df = pd.DataFrame({
        "Container Traffic": [4000],
        "Vessel Traffic": [5],
        "CPI": [243.9],
        "Inflation": [21.5],
        "M2": [2150000],
        "Exchange Rate(USD)": [15.9834]
    })

    new_df["Cargo_lag1"] = df["Cargo Traffic"].iloc[-1]
    new_df["Cargo_lag3"] = df["Cargo Traffic"].iloc[-3]
    new_df["Cargo_lag6"] = df["Cargo Traffic"].iloc[-6]

    new_scaled = scaler.transform(new_df)
    prediction = model.predict(new_scaled)

    results = new_df.copy()
    results["Predicted_Cargo_Traffic"] = prediction

    return results


# -------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------

def main(X_train_scaled, X_test_scaled, y_train, y_test, scaler, df):
    """
    Main training workflow.
    """

    print("\nTraining baseline models...\n")
    baseline_models = train_baseline_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    for name, (model, preds) in baseline_models.items():
        evaluate_model(y_test, preds, name)
        joblib.dump(model, f"models/{name.replace(' ', '_').lower()}.pkl")

    print("\nTraining lag-augmented models...\n")
    lagged_models = train_lagged_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    for name, (model, preds) in lagged_models.items():
        evaluate_model(y_test, preds, name)
        joblib.dump(model, f"models/{name.replace(' ', '_').lower()}.pkl")

    # Example prediction using best-performing lagged RF
    rf_lag_model = lagged_models["Random Forest (Lagged)"][0]
    prediction_results = predict_unseen_data(rf_lag_model, scaler, df)

    print("\nUnseen data prediction:")
    print(prediction_results)


