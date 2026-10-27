"""
================================================================================
 NUMPY DEMONSTRATION
================================================================================
WHAT IS NUMPY?
NumPy ("Numerical Python") is the foundational library for numerical computing
in Python. It provides:
    1. The `ndarray` object -- a fast, memory-efficient, multi-dimensional array
    2. Vectorized math operations (no slow Python `for` loops needed)
    3. Linear algebra, statistics, and random number generation tools

WHY USE IT?
Plain Python lists are slow for numerical work because each element is a full
Python object. NumPy arrays store data in contiguous memory blocks of a single
type, so operations run in optimized, compiled C code -- often 10-100x faster.

PRACTICAL USE CASE IN THIS FILE:
We simulate a small retail business's daily sales data for a week across
3 stores, then use NumPy to:
    - Build the dataset as a 2D array
    - Compute totals, averages, and identify best/worst performers
    - Apply a company-wide 10% discount using vectorized math
    - Generate reproducible random "customer footfall" data
    - Run a matrix operation (useful in real analytics/ML pipelines)

Run this file directly:  python3 01_numpy_demo.py
================================================================================
"""

import numpy as np

print("=" * 70)
print("1. CREATING ARRAYS")
print("=" * 70)

# A Python list vs a NumPy array -- NumPy arrays support element-wise math
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Rows = stores (Store A, Store B, Store C), Columns = days of the week
# This is a realistic 2D dataset: sales revenue ($) per store per day
sales = np.array([
    [1200, 1500, 1100, 1700, 2200, 3000, 2500],   # Store A
    [900,  950,  1000, 1050, 1600, 2100, 1800],   # Store B
    [1400, 1300, 1250, 1600, 2000, 2800, 2600],   # Store C
])

print(f"Sales array shape: {sales.shape}  (3 stores x 7 days)")
print(sales)

print("\n" + "=" * 70)
print("2. AGGREGATE STATISTICS (vectorized -- no manual loops!)")
print("=" * 70)

total_per_store = sales.sum(axis=1)          # sum across each row (per store)
total_per_day = sales.sum(axis=0)            # sum across each column (per day)
avg_per_store = sales.mean(axis=1)
best_day_index = np.argmax(total_per_day)
worst_store_index = np.argmin(total_per_store)

for i, store in enumerate(["Store A", "Store B", "Store C"]):
    print(f"{store}: total=${total_per_store[i]:,}  avg/day=${avg_per_store[i]:,.2f}")

print(f"\nBest sales day across all stores: {days[best_day_index]} "
      f"(${total_per_day[best_day_index]:,})")
print(f"Weakest performing store overall: Store {chr(65 + worst_store_index)}")

print("\n" + "=" * 70)
print("3. VECTORIZED OPERATIONS -- applying a discount instantly")
print("=" * 70)

# Real-world use: apply a 10% weekend discount to ALL numbers at once.
# No loop needed -- this is the core power of NumPy.
discounted_sales = sales * 0.90
print("Sales after a 10% company-wide discount:")
print(np.round(discounted_sales, 2))

print("\n" + "=" * 70)
print("4. RANDOM DATA GENERATION (reproducible with a seed)")
print("=" * 70)

# Real-world use: simulating customer footfall for a forecasting model
rng = np.random.default_rng(seed=42)
footfall = rng.integers(low=50, high=300, size=(3, 7))  # customers per store/day
print("Simulated customer footfall per store per day:")
print(footfall)
print(f"Average footfall across all stores/days: {footfall.mean():.1f}")

print("\n" + "=" * 70)
print("5. LINEAR ALGEBRA -- correlating sales with footfall")
print("=" * 70)

# Real-world use: does more footfall actually correlate with more revenue?
# We flatten both 2D arrays into 1D and compute a correlation matrix.
correlation_matrix = np.corrcoef(sales.flatten(), footfall.flatten())
correlation = correlation_matrix[0, 1]
print(f"Correlation between footfall and sales revenue: {correlation:.3f}")
if correlation > 0.5:
    print("-> Strong positive relationship: more footfall tends to mean more sales.")
else:
    print("-> Weak/no relationship: footfall alone doesn't explain sales well.")

print("\n" + "=" * 70)
print("DONE. Try changing `sales` or the discount rate and re-running!")
print("=" * 70)
