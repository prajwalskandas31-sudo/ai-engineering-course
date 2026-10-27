"""
================================================================================
 PANDAS DEMONSTRATION
================================================================================
WHAT IS PANDAS?
Pandas is Python's core library for working with labeled, tabular data --
think "Excel spreadsheet inside Python." Its two main structures are:
    1. Series    -- a single labeled column of data
    2. DataFrame -- a full table of rows and columns, each column labeled

Pandas is built on top of NumPy, so it inherits NumPy's speed, but adds row
labels, column names, and powerful tools for loading, cleaning, filtering,
grouping, and reshaping real-world messy data.

WHY USE IT?
Raw NumPy arrays are great for pure numbers but have no concept of column
names or mixed data types (e.g., a "Store" column of text next to a "Sales"
column of numbers). Pandas fills that gap and is the standard tool for
loading CSVs/Excel files, cleaning data, and preparing it for analysis or
visualization (this is why Seaborn expects pandas DataFrames as input).

PRACTICAL USE CASE IN THIS FILE:
We continue the same retail scenario used in the NumPy and Matplotlib demos,
but this time as a proper labeled DataFrame (store, day, sales, footfall),
and use Pandas to:
    - Build a tidy DataFrame from raw data
    - Inspect, filter, and sort it
    - Group and aggregate by store and by day
    - Add a computed column (sales per visitor)
    - Handle a realistic messy-data problem: missing values
    - Export the cleaned result to a CSV file

Run this file directly:  python3 04_pandas_demo.py
================================================================================
"""

import numpy as np
import pandas as pd

print("=" * 70)
print("1. BUILDING A TIDY DATAFRAME")
print("=" * 70)

# Same underlying numbers as the NumPy/Matplotlib demos, but reshaped into
# "tidy" format: one row per (store, day) observation -- the format most
# real-world tools (pandas, seaborn, SQL, BI dashboards) expect.
stores = ["Store A", "Store B", "Store C"]
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

sales_grid = np.array([
    [1200, 1500, 1100, 1700, 2200, 3000, 2500],
    [900,  950,  1000, 1050, 1600, 2100, 1800],
    [1400, 1300, 1250, 1600, 2000, 2800, 2600],
])

rng = np.random.default_rng(seed=42)
footfall_grid = rng.integers(low=50, high=300, size=(3, 7))

records = []
for i, store in enumerate(stores):
    for j, day in enumerate(days):
        records.append({
            "store": store,
            "day": day,
            "sales": sales_grid[i, j],
            "footfall": footfall_grid[i, j],
        })

df = pd.DataFrame(records)
print(df.head(10))
print(f"\nDataFrame shape: {df.shape[0]} rows, {df.shape[1]} columns")

print("\n" + "=" * 70)
print("2. INSPECTING THE DATA")
print("=" * 70)

print("Column data types:")
print(df.dtypes)
print("\nQuick statistical summary:")
print(df[["sales", "footfall"]].describe())

print("\n" + "=" * 70)
print("3. FILTERING AND SORTING")
print("=" * 70)

# Real-world use: "show me only weekend sales over $2000, best first"
weekend_high_sales = df[(df["day"].isin(["Sat", "Sun"])) & (df["sales"] > 2000)]
weekend_high_sales = weekend_high_sales.sort_values("sales", ascending=False)
print("Weekend days with sales over $2000:")
print(weekend_high_sales)

print("\n" + "=" * 70)
print("4. GROUPING AND AGGREGATING")
print("=" * 70)

# Real-world use: total and average sales per store (like a pivot table)
store_summary = df.groupby("store")["sales"].agg(["sum", "mean", "max"])
store_summary.columns = ["total_sales", "avg_sales", "best_day_sales"]
print("Summary by store:")
print(store_summary.round(2))

# Real-world use: which day of the week performs best across all stores?
day_summary = df.groupby("day")["sales"].sum().sort_values(ascending=False)
print("\nTotal sales by day (best to worst):")
print(day_summary)

print("\n" + "=" * 70)
print("5. ADDING A COMPUTED COLUMN")
print("=" * 70)

# Real-world use: sales per visitor -- reveals which store converts
# foot traffic into revenue most efficiently, not just which is busiest.
df["sales_per_visitor"] = (df["sales"] / df["footfall"]).round(2)
print(df.sort_values("sales_per_visitor", ascending=False).head(5))

best_row = df.loc[df["sales_per_visitor"].idxmax()]
print(f"\nMost efficient store/day: {best_row['store']} on {best_row['day']} "
      f"(${best_row['sales_per_visitor']} per visitor)")

print("\n" + "=" * 70)
print("6. HANDLING MISSING DATA (a common real-world problem)")
print("=" * 70)

# Simulate a sensor outage: footfall wasn't recorded for a few rows
df_with_gaps = df.copy()
df_with_gaps.loc[[2, 9, 15], "footfall"] = np.nan
print(f"Missing values before cleaning:\n{df_with_gaps.isna().sum()}")

# Fill missing footfall with that store's average footfall (a realistic,
# defensible strategy -- better than dropping rows or using a global average)
df_with_gaps["footfall"] = df_with_gaps.groupby("store")["footfall"] \
    .transform(lambda x: x.fillna(x.mean()))
print(f"\nMissing values after filling with per-store average:\n{df_with_gaps.isna().sum()}")

print("\n" + "=" * 70)
print("7. EXPORTING RESULTS")
print("=" * 70)

output_path = "sales_summary.csv"
store_summary.to_csv(output_path)
print(f"Store summary exported to: {output_path}")

print("\nSummary of what this script demonstrated:")
print(" - pd.DataFrame()      -> building tidy, labeled tabular data")
print(" - boolean filtering   -> selecting rows that meet conditions")
print(" - .sort_values()      -> ordering results")
print(" - .groupby().agg()    -> pivot-table-style summaries")
print(" - computed columns    -> deriving new metrics (sales per visitor)")
print(" - .fillna()           -> handling missing/messy real-world data")
print(" - .to_csv()           -> exporting cleaned results for other tools")
