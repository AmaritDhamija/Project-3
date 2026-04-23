"""
Helper functions for Winnipeg Crime Analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    """
    Load the crime dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)


def clean_data(df):
    """
    Clean the dataset by removing rows with missing location values
    and standardizing column names.

    Parameters:
        df (pd.DataFrame): Raw crime dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    df = df.copy()

    df = df.dropna(subset=["Neighbourhood", "Community"])

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

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