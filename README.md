"""
Customer Data Analysis & Insights
----------------------------------
Loads raw customer data, cleans and preprocesses it, runs exploratory
analysis, and outputs summary visualizations and a bottleneck report.

Usage:
    python customer_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "customer_data.csv"
OUTPUT_DIR = "outputs"


def load_data(path: str) -> pd.DataFrame:
    """Load raw customer data from CSV."""
    df = pd.read_csv(path, parse_dates=["signup_date"])
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the raw dataset.

    - Removes exact duplicate rows
    - Fills missing numeric values with the column median
    - Fills missing categorical values with 'Unknown'
    - Drops rows with invalid customer IDs
    """
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows.")

    df = df[df["customer_id"].notna()]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing = df[col].isna().sum()
        if missing:
            df[col] = df[col].fillna(df[col].median())
            print(f"Filled {missing} missing values in '{col}' with median.")

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        missing = df[col].isna().sum()
        if missing:
            df[col] = df[col].fillna("Unknown")
            print(f"Filled {missing} missing values in '{col}' with 'Unknown'.")

    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer a few simple features used in the analysis."""
    df["tenure_days"] = (pd.Timestamp.today() - df["signup_date"]).dt.days
    df["is_active"] = df["orders_last_90d"] > 0
    df["ticket_rate"] = df["support_tickets"] / df["orders_last_90d"].replace(0, 1)
    return df


def exploratory_summary(df: pd.DataFrame) -> None:
    """Print key summary statistics and trends."""
    print("\n--- Spend by Segment ---")
    print(df.groupby("segment")["monthly_spend"].agg(["mean", "median", "count"]).round(2))

    print("\n--- Spend by Region ---")
    print(df.groupby("region")["monthly_spend"].mean().round(2).sort_values(ascending=False))

    print("\n--- Satisfaction vs Support Tickets (correlation) ---")
    corr = df["satisfaction_score"].corr(df["support_tickets"])
    print(f"Correlation: {corr:.2f}")

    print("\n--- Inactive Customers (0 orders in last 90 days) ---")
    inactive_pct = (~df["is_active"]).mean() * 100
    print(f"{inactive_pct:.1f}% of customers had no orders in the last 90 days.")


def identify_bottlenecks(df: pd.DataFrame) -> pd.DataFrame:
    """Flag segments/regions with high support-ticket rates or low satisfaction,
    which is where operational bottlenecks tend to show up."""
    summary = (
        df.groupby(["region", "segment"])
        .agg(
            avg_satisfaction=("satisfaction_score", "mean"),
            avg_ticket_rate=("ticket_rate", "mean"),
            customers=("customer_id", "count"),
        )
        .round(2)
        .reset_index()
    )
    bottlenecks = summary[
        (summary["avg_satisfaction"] < summary["avg_satisfaction"].median())
        & (summary["avg_ticket_rate"] > summary["avg_ticket_rate"].median())
    ].sort_values("avg_ticket_rate", ascending=False)

    print("\n--- Potential Bottleneck Segments (low satisfaction, high ticket rate) ---")
    print(bottlenecks.to_string(index=False))
    return bottlenecks


def make_visualizations(df: pd.DataFrame, outdir: str) -> None:
    """Save a couple of key charts to the output folder."""
    import os
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(7, 4))
    df.groupby("segment")["monthly_spend"].mean().plot(kind="bar", color="#1F3864")
    plt.title("Average Monthly Spend by Segment")
    plt.ylabel("Avg Monthly Spend")
    plt.tight_layout()
    plt.savefig(f"{outdir}/spend_by_segment.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    df.groupby("region")["satisfaction_score"].mean().plot(kind="bar", color="#2E5B9A")
    plt.title("Average Satisfaction Score by Region")
    plt.ylabel("Avg Satisfaction (1-5)")
    plt.tight_layout()
    plt.savefig(f"{outdir}/satisfaction_by_region.png")
    plt.close()

    print(f"\nSaved charts to '{outdir}/'.")


def main():
    df = load_data(INPUT_FILE)
    df = clean_data(df)
    df = add_features(df)
    exploratory_summary(df)
    identify_bottlenecks(df)
    make_visualizations(df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
