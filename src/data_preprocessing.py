"""
Data preprocessing module for Tema Port cargo throughput project.

This script:
- Loads the merged dataset
- Cleans column names
- Converts and sorts date column
- Performs basic data inspection
- Saves the cleaned dataset for downstream modeling
"""

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the raw dataset from an Excel file.

    Parameters
    ----------
    file_path : str
        Path to the raw Excel dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    return pd.read_excel(file_path)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with cleaned column names.
    """
    df.columns = df.columns.str.strip()
    return df


def process_date_column(df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    """
    Convert date column to datetime format and sort the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    date_column : str, optional
        Name of the date column, by default "Date".

    Returns
    -------
    pd.DataFrame
        Dataset with processed and sorted date column.
    """
    df[date_column] = df[date_column].astype(str).str.replace("M", "-", regex=False)
    df[date_column] = pd.to_datetime(df[date_column], format="%Y-%m")
    df = df.sort_values(date_column).reset_index(drop=True)
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Perform basic data inspection and print summaries.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to inspect.
    """
    print("\nMissing values summary:")
    print(df.isnull().sum())

    print("\nDataset info:")
    print(df.info())

    print("\nDescriptive statistics:")
    print(df.describe())


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned dataset to CSV format.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.
    output_path : str
        Path to save the processed CSV file.
    """
    df.to_csv(output_path, index=False)


def main() -> None:
    """
    Main execution function for data preprocessing.
    """
    raw_data_path = "data/raw/MERGED DATASET.xlsx"
    processed_data_path = "data/processed/tema_port_processed.csv"

    df = load_data(raw_data_path)
    df = clean_columns(df)
    df = process_date_column(df)

    inspect_data(df)
    save_processed_data(df, processed_data_path)

    print("\nData preprocessing completed successfully.")
    print(f"Processed data saved to: {processed_data_path}")


if __name__ == "__main__":
    main()






































































