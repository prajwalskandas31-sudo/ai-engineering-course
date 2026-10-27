# Matplotlib — End-to-End Use Case Explanation

**Companion code file:** `02_matplotlib_demo.py`
**Output produced:** `sales_dashboard.png`
**Scenario:** Turn the same NumPy retail sales dataset into a visual dashboard a store manager could actually read and act on.

---

## 1. Why Matplotlib Is Used Here

Numbers in a table (like the raw `sales` array) are hard to interpret at a glance — you can't "see" a trend or an outlier in a grid of digits. Matplotlib converts numeric arrays directly into charts, and because it works natively with NumPy arrays, no data conversion is needed between the two libraries. This script builds one **figure** containing four different chart types, mimicking a real business dashboard.

---

## 2. Step-by-Step Walkthrough

### Step 1 — Setting Up a Multi-Panel Figure
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Weekly Retail Sales Dashboard", fontsize=16, fontweight="bold")
```
- `plt.subplots(2, 2, ...)` creates one **figure** (`fig`, the overall canvas) containing a **2×2 grid of axes** (`axes`, the individual chart panels) — this is the standard Matplotlib pattern for dashboards.
- `axes` is a 2D NumPy-like array itself, so `axes[0, 0]` is the top-left panel, `axes[1, 1]` is the bottom-right, and so on — indexing works exactly like a NumPy array.
- `fig.suptitle` adds one title over the entire dashboard, distinct from each panel's own `set_title`.

### Step 2 — Line Chart: Daily Sales Trend (`ax.plot`)
```python
for i, store in enumerate(stores):
    ax1.plot(days, sales[i], marker="o", label=store)
```
- Loops through each store's row of the `sales` array and plots it as its own line, with a marker at each data point.
- `label=store` feeds into `ax1.legend()` so each line is identifiable.
- **When to use a line chart:** whenever the x-axis represents an ordered sequence (here, days of the week) and you want to see the *shape* of change over time — rising, falling, spikes on weekends, etc.

### Step 3 — Bar Chart: Total Weekly Revenue (`ax.bar`)
```python
ax2.bar(stores, total_per_store, color=bar_colors)
for i, val in enumerate(total_per_store):
    ax2.text(i, val + 100, f"${val:,}", ha="center", fontweight="bold")
```
- Bars compare a single summary number (total weekly revenue) across discrete categories (the 3 stores) — bar charts are the standard choice whenever you're comparing categories rather than showing a trend.
- The `ax2.text(...)` loop manually places a dollar-value label above each bar — a common finishing touch that makes a chart "presentation-ready" rather than just a rough plot.

### Step 4 — Pie Chart: Revenue Share (`ax.pie`)
```python
ax3.pie(total_per_store, labels=stores, autopct="%1.1f%%", colors=bar_colors, startangle=90)
```
- Pie charts answer a different question than bar charts: not "how big is each value?" but "what **proportion of the whole** does each value represent?"
- `autopct="%1.1f%%"` automatically computes and displays each slice's percentage of the total — you don't need to calculate percentages yourself.
- **Caution worth noting:** pie charts work well for a small number of categories (here, 3 stores); they get hard to read with many slices, which is why bar charts are usually preferred beyond 5–6 categories.

### Step 5 — Scatter Plot: Footfall vs. Sales (`ax.scatter`)
```python
ax4.scatter(footfall.flatten(), sales.flatten(), c="#C44E52", alpha=0.8)
```
- Each point represents one (footfall, sales) pair from one store on one day — 21 points total, matching the correlation calculated in the NumPy script.
- Scatter plots are the standard way to visually confirm what a correlation coefficient (like the `0.131` computed with `np.corrcoef`) actually looks like — here, the widely scattered cloud of points visually confirms the weak relationship found numerically.
- `alpha=0.8` adds slight transparency so overlapping points remain distinguishable.

### Step 6 — Layout and Export
```python
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("sales_dashboard.png", dpi=150)
```
- `tight_layout` automatically adjusts spacing so titles and labels don't overlap between the 4 panels; the `rect` argument reserves space at the top for `suptitle`.
- `savefig(..., dpi=150)` exports the entire figure as a high-resolution PNG suitable for embedding in a report or email — `plt.show()` (commented out) would instead open an interactive window, which only works in a local/desktop environment, not headless scripts.

---

## 3. End-to-End Takeaway

| Panel | Chart Type | Matplotlib Call | Business Question Answered |
|---|---|---|---|
| Top-left | Line | `ax.plot()` | How did sales trend across the week? |
| Top-right | Bar | `ax.bar()` | Which store earned the most total revenue? |
| Bottom-left | Pie | `ax.pie()` | What share of total revenue does each store hold? |
| Bottom-right | Scatter | `ax.scatter()` | Does more foot traffic actually mean more sales? |

The core lesson: **choice of chart type should match the question being asked** — trend → line, comparison → bar, proportion → pie, relationship → scatter. Matplotlib gives full manual control over all of them within a single reusable dashboard layout.
