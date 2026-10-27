"""
================================================================================
 SEABORN DEMONSTRATION
================================================================================
WHAT IS SEABORN?
Seaborn is a statistical data visualization library built on top of
Matplotlib. It integrates tightly with pandas DataFrames and adds:
    1. Attractive default styling (less manual formatting than raw Matplotlib)
    2. Built-in statistical plots: distributions, regressions, categorical
       comparisons, and correlation heatmaps
    3. Automatic handling of grouping/color-coding by category

WHY USE IT?
When your data is naturally tabular (rows = observations, columns = variables,
like a spreadsheet), Seaborn lets you produce publication-quality statistical
plots in one line instead of many lines of Matplotlib boilerplate.

PRACTICAL USE CASE IN THIS FILE:
We use Seaborn's classic built-in "tips" dataset -- real-world-style data
from a restaurant recording each bill: total_bill, tip, gender, smoker
status, day, time, and party size. This mirrors a real business analytics
task: understanding what drives customer tipping behavior.

We build:
    1. A histogram/KDE of tip amounts (distribution analysis)
    2. A scatter+regression plot of total_bill vs tip (trend analysis)
    3. A box plot of tips by day (categorical comparison)
    4. A correlation heatmap of the numeric columns (relationship overview)

The final dashboard is saved as 'tips_dashboard.png'.

Run this file directly:  python3 03_seaborn_demo.py
================================================================================
"""

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")  # Seaborn's clean default aesthetic

print("=" * 70)
print("1. LOADING A REAL-WORLD-STYLE DATASET")
print("=" * 70)

# Seaborn ships with several sample datasets useful for learning/demoing.
# 'tips' = restaurant billing records (loaded as a pandas DataFrame)
tips = sns.load_dataset("tips")
print(tips.head())
print(f"\nDataset shape: {tips.shape[0]} rows, {tips.shape[1]} columns")
print(f"Columns: {list(tips.columns)}")

print("\n" + "=" * 70)
print("2. QUICK STATISTICAL SUMMARY")
print("=" * 70)
print(tips[["total_bill", "tip", "size"]].describe())

# --------------------------------------------------------------------------
# Build a 2x2 dashboard of statistical plots
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Restaurant Tipping Behavior Dashboard", fontsize=16, fontweight="bold")

# 1. DISTRIBUTION PLOT -- shape of the tip amounts
sns.histplot(data=tips, x="tip", kde=True, color="#4C72B0", ax=axes[0, 0])
axes[0, 0].set_title("Distribution of Tip Amounts")

# 2. REGRESSION PLOT -- does a bigger bill mean a bigger tip?
sns.regplot(data=tips, x="total_bill", y="tip", color="#DD8452", ax=axes[0, 1],
            scatter_kws={"alpha": 0.5})
axes[0, 1].set_title("Total Bill vs Tip (with trend line)")

# 3. BOX PLOT -- do tips vary by day of the week?
sns.boxplot(data=tips, x="day", y="tip", hue="day", palette="Set2",
            legend=False, ax=axes[1, 0])
axes[1, 0].set_title("Tip Amount by Day of Week")

# 4. HEATMAP -- correlation between numeric variables
numeric_cols = tips[["total_bill", "tip", "size"]]
corr = numeric_cols.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1, 1])
axes[1, 1].set_title("Correlation Heatmap")

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = "tips_dashboard.png"
plt.savefig(output_path, dpi=150)
print(f"\nDashboard saved to: {output_path}")

print("\n" + "=" * 70)
print("3. KEY INSIGHTS PULLED DIRECTLY FROM THE DATA")
print("=" * 70)
best_tipping_day = tips.groupby("day", observed=True)["tip"].mean().idxmax()
avg_tip_pct = (tips["tip"] / tips["total_bill"] * 100).mean()
correlation = tips["total_bill"].corr(tips["tip"])

print(f"Day with the highest average tip: {best_tipping_day}")
print(f"Average tip as % of bill: {avg_tip_pct:.1f}%")
print(f"Correlation between bill size and tip: {correlation:.2f} "
      f"({'strong' if correlation > 0.5 else 'moderate/weak'} relationship)")

print("\nSummary of what each subplot demonstrates:")
print(" - Top-left:    sns.histplot() -> distribution/shape of one variable")
print(" - Top-right:   sns.regplot()  -> relationship + trend between 2 variables")
print(" - Bottom-left: sns.boxplot()  -> comparing a variable across categories")
print(" - Bottom-right sns.heatmap()  -> correlation across all numeric variables")
