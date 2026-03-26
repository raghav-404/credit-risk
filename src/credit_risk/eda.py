import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_class_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='target')
    plt.title("Class Distribution: 0 (Good Loan) vs 1 (Default)")
    plt.show()


def plot_feature_correlations(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=np.number)
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlations")
    plt.show()
