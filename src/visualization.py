import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_sale_price_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["SalePrice"],
        bins=30,
        kde=True
    )

    plt.title("Distribution of House Sale Prices")
    plt.xlabel("Sale Price")
    plt.ylabel("Number of Houses")
    plt.tight_layout()
    plt.show()


def plot_living_area_vs_price(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="SqFtTotLiving",
        y="SalePrice",
        alpha=0.6
    )

    plt.title("Living Area vs House Sale Price")
    plt.xlabel("Total Living Area (sq ft)")
    plt.ylabel("Sale Price")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 10))

    correlation = df.select_dtypes(
        include="number"
    ).corr()

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def plot_yearly_average_price(df: pd.DataFrame) -> None:
    yearly_price = (
        df.groupby("SaleYear")["SalePrice"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=yearly_price,
        x="SaleYear",
        y="SalePrice",
        marker="o"
    )

    plt.title("Average House Sale Price by Year")
    plt.xlabel("Sale Year")
    plt.ylabel("Average Sale Price")
    plt.tight_layout()
    plt.show()


def plot_price_by_building_grade(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="BldgGrade",
        y="SalePrice"
    )

    plt.title("Sale Price by Building Grade")
    plt.xlabel("Building Grade")
    plt.ylabel("Sale Price")
    plt.tight_layout()
    plt.show()


def plot_bedroom_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))

    sns.countplot(
        data=df,
        x="Bedrooms"
    )

    plt.title("Distribution of Bedrooms")
    plt.xlabel("Number of Bedrooms")
    plt.ylabel("Number of Houses")
    plt.tight_layout()
    plt.show()


def plot_building_grade_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))

    sns.countplot(
        data=df,
        x="BldgGrade"
    )

    plt.title("Distribution of Building Grades")
    plt.xlabel("Building Grade")
    plt.ylabel("Number of Houses")
    plt.tight_layout()
    plt.show()