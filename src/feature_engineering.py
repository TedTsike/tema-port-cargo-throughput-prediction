"""
Feature engineering module for Tema Port cargo throughput project.

This script:
- Loads the processed dataset
- Performs exploratory visualizations
- Conducts stationarity tests
- Creates lagged features
- Splits data into training and testing sets
- Scales features for machine learning models
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, kpss


def load_processed_data(file_path: str) -> pd.DataFrame:
    """
    Load the cleaned and processed dataset.

    Parameters
    ----------
    file_path : str
        Path to the processed CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    return pd.read_csv(file_path, parse_dates=["Date"])


def plot_cargo_throughput(df: pd.DataFrame) -> None:
    """
    Plot monthly cargo throughput over time.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing cargo throughput data.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(df["Date"], df["Cargo Traffic"])
    plt.title("Monthly Cargo Throughput at Tema Port")
    plt.xlabel("Year")
    plt.ylabel("Cargo Throughput")

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Plot correlation matrix of cargo throughput and predictor variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing numeric variables.
    """
    corr = df.drop(columns=["Date"]).corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix of Cargo Throughput and Predictors")
    plt.tight_layout()
    plt.show()


def plot_predictor_relationships(df: pd.DataFrame, predictors: list) -> None:
    """
    Plot scatter relationships between cargo throughput and predictors.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing predictors.
    predictors : list
        List of predictor variable names.
    """
    for var in predictors:
        sns.scatterplot(x=df[var], y=df["Cargo Traffic"])
        plt.title(f"Cargo Throughput vs {var}")
        plt.xlabel(var)
        plt.ylabel("Cargo Throughput")
        plt.tight_layout()
        plt.show()


def stationarity_tests(series: pd.Series) -> None:
    """
    Perform ADF and KPSS stationarity tests on cargo throughput.

    Parameters
    ----------
    series : pd.Series
        Time series to test.
    """
    adf_result = adfuller(series)
    print("ADF Statistic:", adf_result[0])
    print("ADF p-value:", adf_result[1])

    kpss_result = kpss(series, regression="c", nlags="auto")
    print("KPSS Statistic:", kpss_result[0])
    print("KPSS p-value:", kpss_result[1])


def create_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Create final feature matrix with lagged variables.

    Features included:
    - Container Traffic: containerized cargo volume
    - Vessel Traffic: number of vessel calls
    - CPI: consumer price index (economic activity)
    - Inflation: inflation rate (macroeconomic pressure)
    - M2: money supply (liquidity indicator)
    - Exchange Rate (USD): currency valuation impact

    Lagged features (1, 3, 6 months) capture delayed effects
    of operational and macroeconomic variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    tuple
        Feature matrix X and target vector y.
    """
    y = df["Cargo Traffic"]

    X = df[
        [
            "Container Traffic",
            "Vessel Traffic",
            "CPI",
            "Inflation",
            "M2",
            "Exchange Rate(USD)"
        ]
    ]

    for lag in [1, 3, 6]:
        X[f"Container_lag{lag}"] = df["Container Traffic"].shift(lag)
        X[f"Vessel_lag{lag}"] = df["Vessel Traffic"].shift(lag)
        X[f"M2_lag{lag}"] = df["M2"].shift(lag)
        X[f"Inflation_lag{lag}"] = df["Inflation"].shift(lag)

    X = X.dropna()
    y = y.loc[X.index]

    return X, y


def split_and_scale_features(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Split dataset into training and testing sets and apply standard scaling.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.

    Returns
    -------
    tuple
        Scaled training and testing sets and fitted scaler.
    """
    split_index = int(len(X) * 0.8)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def main() -> None:
    """
    Main execution function for feature engineering.
    """
    processed_data_path = "data/processed/tema_port_processed.csv"

    predictors = [
        "Container Traffic",
        "Vessel Traffic",
        "M2",
        "Exchange Rate(USD)",
        "Inflation",
        "CPI"
    ]

    df = load_processed_data(processed_data_path)

    plot_cargo_throughput(df)
    plot_correlation_matrix(df)
    plot_predictor_relationships(df, predictors)
    stationarity_tests(df["Cargo Traffic"])

    X, y = create_feature_matrix(df)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_features(X, y)

    print("Feature engineering completed successfully.")
    print(f"Final feature matrix shape: {X.shape}")


if __name__ == "__main__":
    main()
