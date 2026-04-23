"""
Helper functions for Winnipeg Crime Analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    """
    Load the COVID-19 by-law enforcement dataset from a CSV file.

    Parameters:
        file_path (str): Relative or absolute path to the CSV file.

    Returns:
        pd.DataFrame: Raw dataset loaded into a pandas DataFrame.
    """
    return pd.read_csv(file_path)


def clean_data(df):
    """
    Clean COVID enforcement dataset.

    Steps:
    - convert date column
    - fix numeric columns
    - handle missing values
    - clean column names
    """

    df = df.copy()

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop rows with invalid dates
    df = df.dropna(subset=["Date"])

    # Convert numeric columns that are wrongly stored as object
    cols_to_numeric = [
        "Number of positive interactions",
        "Total Number of interactions"
    ]

    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing numeric values with 0
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )

    return df

    return df


def summarize_by_group(df, group_col, value_col):
    """
    Group the dataset by one column and sum another column.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.
        group_col (str): Column to group by.
        value_col (str): Column to aggregate.

    Returns:
        pd.Series: Grouped and summed results sorted descending.
    """
    return (
        df.groupby(group_col)[value_col]
        .sum()
        .sort_values(ascending=False)
    )


def plot_yearly_trend(df):
    """
    Plot total crime counts by year.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        None
    """
    yearly = df.groupby("year")["count_stats"].sum()

    fig, ax = plt.subplots()
    yearly.plot(ax=ax)
    ax.set_title("Total Crime Count by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Crime Count")
    plt.show()


def plot_crime_type_distribution(df):
    """
    Plot the distribution of crime categories.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        None
    """
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="crimetype_inc", ax=ax)
    ax.set_title("Crime Type Distribution")
    ax.tick_params(axis="x", rotation=45)
    plt.show()