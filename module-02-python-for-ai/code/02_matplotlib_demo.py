"""
================================================================================
 MATPLOTLIB DEMONSTRATION
================================================================================
WHAT IS MATPLOTLIB?
Matplotlib is Python's foundational plotting library. It gives you fine-grained
control to build almost any kind of static, animated, or interactive chart:
line plots, bar charts, scatter plots, histograms, pie charts, and more.

WHY USE IT?
Raw numbers are hard to interpret. Visualizing data reveals trends, outliers,
and relationships instantly. Matplotlib is the base layer most other Python
plotting tools (including Seaborn) are built on top of.

PRACTICAL USE CASE IN THIS FILE:
Continuing the retail sales scenario, we use Matplotlib to build a
multi-panel dashboard (a "figure" with subplots) that a manager could
actually use to review weekly performance:
    1. Line chart -- daily sales trend per store (spot patterns over time)
    2. Bar chart  -- total weekly revenue per store (compare performance)
    3. Pie chart  -- revenue share per store (see proportions at a glance)
    4. Scatter plot -- footfall vs. sales (does more traffic mean more $?)

The final dashboard is saved as 'sales_dashboard.png' so it can be shared
in a report or email.

Run this file directly:  python3 02_matplotlib_demo.py
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ---- Reuse the same sample dataset as the NumPy demo ----
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
stores = ["Store A", "Store B", "Store C"]

sales = np.array([
    [1200, 1500, 1100, 1700, 2200, 3000, 2500],   # Store A
    [900,  950,  1000, 1050, 1600, 2100, 1800],   # Store B
    [1400, 1300, 1250, 1600, 2000, 2800, 2600],   # Store C
])

rng = np.random.default_rng(seed=42)
footfall = rng.integers(low=50, high=300, size=(3, 7))

total_per_store = sales.sum(axis=1)

# --------------------------------------------------------------------------
# Build a 2x2 dashboard figure -- a very common real-world Matplotlib layout
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Weekly Retail Sales Dashboard", fontsize=16, fontweight="bold")

# 1. LINE CHART -- trend of each store's sales across the week
ax1 = axes[0, 0]
for i, store in enumerate(stores):
    ax1.plot(days, sales[i], marker="o", label=store)
ax1.set_title("Daily Sales Trend")
ax1.set_xlabel("Day of Week")
ax1.set_ylabel("Sales ($)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. BAR CHART -- total weekly revenue comparison
ax2 = axes[0, 1]
bar_colors = ["#4C72B0", "#DD8452", "#55A868"]
ax2.bar(stores, total_per_store, color=bar_colors)
ax2.set_title("Total Weekly Revenue by Store")
ax2.set_ylabel("Revenue ($)")
for i, val in enumerate(total_per_store):
    ax2.text(i, val + 100, f"${val:,}", ha="center", fontweight="bold")

# 3. PIE CHART -- revenue share of each store
ax3 = axes[1, 0]
ax3.pie(
    total_per_store,
    labels=stores,
    autopct="%1.1f%%",
    colors=bar_colors,
    startangle=90,
)
ax3.set_title("Revenue Share by Store")

# 4. SCATTER PLOT -- footfall vs sales relationship
ax4 = axes[1, 1]
ax4.scatter(footfall.flatten(), sales.flatten(), c="#C44E52", alpha=0.8)
ax4.set_title("Customer Footfall vs Sales")
ax4.set_xlabel("Footfall (customers)")
ax4.set_ylabel("Sales ($)")
ax4.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave room for suptitle

output_path = "sales_dashboard.png"
plt.savefig(output_path, dpi=150)
print(f"Dashboard saved to: {output_path}")

# NOTE: plt.show() opens an interactive window -- useful when running
# locally, but commented out here since this script may run headless.
# plt.show()

print("\nSummary of what each subplot demonstrates:")
print(" - Top-left:    ax.plot()   -> trends over time")
print(" - Top-right:   ax.bar()    -> categorical comparisons")
print(" - Bottom-left: ax.pie()    -> proportions of a whole")
print(" - Bottom-right ax.scatter()-> relationship between two variables")
