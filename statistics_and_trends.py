"""
Statistics and Trends Assignment - Heart Disease Dataset

This script performs preprocessing, statistical analysis, and visualizations
on the heart disease dataset provided (heart.csv). It follows PEP-8 standards
and includes complete documentation and output formatting.
"""

# from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """
    Plot a relational scatter plot showing the relationship between
    cholesterol and maximum heart rate (MaxHR) by heart disease status.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Cholesterol', y='MaxHR', hue='HeartDisease', data=df, ax=ax)
    ax.set_title('Cholesterol vs Max Heart Rate by Heart Disease')
    plt.savefig('relational_plot.png')
    plt.close(fig)
    return


def plot_categorical_plot(df):
    """
    Plot a categorical bar chart of average resting blood pressure (RestingBP)
    across different chest pain types, colored by heart disease presence.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='ChestPainType', y='RestingBP', hue='HeartDisease', data=df, ax=ax, errorbar=None)
    ax.set_title('Average Resting Blood Pressure by Chest Pain Type')
    plt.savefig('categorical_plot.png')
    plt.close(fig)
    return


def plot_statistical_plot(df):
    """
    Plot a histogram and kernel density estimate of cholesterol levels.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['Cholesterol'], kde=True, color='teal', ax=ax)
    ax.set_title('Distribution of Cholesterol Levels')
    plt.savefig('statistical_plot.png')
    plt.close(fig)
    return


def statistical_analysis(df, col: str):
    """
    Compute basic statistical moments (mean, std, skewness, kurtosis)
    for the specified numeric column.
    """
    series = df[col].dropna()
    mean = series.mean()
    stddev = series.std()
    skew = ss.skew(series)
    excess_kurtosis = ss.kurtosis(series)
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Perform basic preprocessing: display statistics, handle missing values,
    and ensure correct data types.
    """
    print("Initial data overview:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())

    # Convert categorical indicators (Y/N, M/F) to numeric where appropriate
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})

    # Remove rows with missing or invalid cholesterol values
    df = df[df['Cholesterol'] > 0].dropna()

    print("\nCorrelation matrix:")
    print(df.corr(numeric_only=True))
    return df


def writing(moments, col):
    """
    Print the computed statistical moments and interpret their meaning.
    """
    print(f'\nFor the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')

    if moments[2] > 0.5:
        skew_desc = "right skewed"
    elif moments[2] < -0.5:
        skew_desc = "left skewed"
    else:
        skew_desc = "approximately symmetrical"

    if moments[3] > 0:
        kurt_desc = "leptokurtic (heavy tails)"
    elif moments[3] < 0:
        kurt_desc = "platykurtic (light tails)"
    else:
        kurt_desc = "mesokurtic (normal-like)"

    print(f"The data distribution is {skew_desc} and {kurt_desc}.")
    return


def main():
    """
    Run the full analysis pipeline:
    - Load dataset
    - Preprocess data
    - Plot visualizations
    - Compute and report statistical moments
    """
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'Cholesterol'

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == '__main__':
    main()
