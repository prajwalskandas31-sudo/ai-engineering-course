# NumPy — End-to-End Use Case Explanation

**Companion code file:** `01_numpy_demo.py`
**Scenario:** A retail business wants to analyze one week of sales data across 3 stores, apply a discount, simulate customer footfall, and check whether footfall drives revenue.

---

## 1. Why NumPy Is Used Here

Retail analytics involves grids of numbers — stores × days, revenue × footfall — and requires fast math across all of them at once. Python lists would force manual `for` loops for every calculation. NumPy's `ndarray` stores numbers in contiguous memory and lets you operate on entire rows, columns, or the whole array in a single vectorized call, which is both faster and far less code.

---

## 2. Step-by-Step Walkthrough

### Step 1 — Building the Dataset (`np.array`)
```python
sales = np.array([
    [1200, 1500, 1100, 1700, 2200, 3000, 2500],   # Store A
    [900,  950,  1000, 1050, 1600, 2100, 1800],   # Store B
    [1400, 1300, 1250, 1600, 2000, 2800, 2600],   # Store C
])
```
- This creates a **2D array** with shape `(3, 7)` — 3 rows (stores), 7 columns (days).
- **Real-world parallel:** this is exactly how you'd represent a spreadsheet of daily revenue per store, except NumPy lets you treat the whole grid as one mathematical object instead of nested lists.
- `sales.shape` confirms the dimensions, which matters when you later combine this array with others (e.g., footfall) — shapes must be compatible for element-wise operations.

### Step 2 — Aggregate Statistics (`sum`, `mean`, `argmax`, `argmin`)
```python
total_per_store = sales.sum(axis=1)   # sum across each row
total_per_day   = sales.sum(axis=0)   # sum across each column
avg_per_store   = sales.mean(axis=1)
```
- `axis=1` collapses **columns** (days) → one total per store (row-wise sum).
- `axis=0` collapses **rows** (stores) → one total per day (column-wise sum).
- This axis logic is the single most important NumPy concept for tabular data — getting it backwards silently gives you the wrong answer, so it's worth memorizing: **axis=0 moves down rows, axis=1 moves across columns.**
- `np.argmax(total_per_day)` returns the *index* of the best day, not the value itself — this index is then used to look up the human-readable day name (`days[best_day_index]`), a common pattern when connecting numeric results back to labels.
- **Business value:** in one line each, we know total revenue per store, the daily average, the single best sales day of the week, and the weakest-performing store — the kind of summary a manager would want in a weekly report.

### Step 3 — Vectorized Discounting
```python
discounted_sales = sales * 0.90
```
- Multiplying an entire array by a scalar applies the operation to *every single element* simultaneously — this is called **broadcasting**.
- **Real-world parallel:** modeling "what if we ran a 10% weekend discount storewide?" without writing a loop over 21 individual numbers. This same pattern scales to millions of rows in real financial models.

### Step 4 — Reproducible Random Data (`np.random.default_rng`)
```python
rng = np.random.default_rng(seed=42)
footfall = rng.integers(low=50, high=300, size=(3, 7))
```
- `default_rng` is NumPy's modern random number generator (preferred over the legacy `np.random.rand`).
- The `seed=42` makes the "random" numbers **reproducible** — anyone who runs this script gets identical footfall numbers, which matters for debugging and for sharing reproducible analysis with colleagues.
- **Real-world parallel:** simulating customer counts when you don't yet have a footfall sensor installed, so you can prototype a forecasting model before real data arrives.

### Step 5 — Correlation via Linear Algebra (`np.corrcoef`)
```python
correlation_matrix = np.corrcoef(sales.flatten(), footfall.flatten())
correlation = correlation_matrix[0, 1]
```
- `.flatten()` converts each 2D array (3×7) into a 1D array of 21 values, so the two datasets can be compared point-by-point.
- `np.corrcoef` returns a 2×2 **correlation matrix**; position `[0, 1]` is the correlation between the two input arrays (the diagonal `[0,0]` and `[1,1]` are always 1.0, since anything correlates perfectly with itself).
- A value near **+1** means footfall strongly predicts sales; near **0** means they're unrelated; negative would mean an inverse relationship.
- **Business value:** this single number answers a real strategic question — "should we invest in driving foot traffic to increase sales, or is something else (like average spend per visit) the bigger lever?"

---

## 3. End-to-End Takeaway

| Step | NumPy Feature | Business Question Answered |
|---|---|---|
| Build dataset | `np.array` (2D) | How is our raw data structured? |
| Aggregate | `.sum()`, `.mean()`, `axis` | Which store/day performs best or worst? |
| Transform | Broadcasting (`* 0.90`) | What happens if we apply a discount everywhere? |
| Simulate | `default_rng().integers()` | What if we don't have real data yet — can we prototype? |
| Relate | `np.corrcoef` | Does variable X actually influence variable Y? |

This mirrors a realistic analytics workflow: **load → summarize → transform → simulate → relate** — the same five steps apply whether the numbers are retail sales, sensor readings, or financial transactions.
