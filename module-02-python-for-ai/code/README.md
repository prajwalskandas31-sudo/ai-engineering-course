# README — Python Data Libraries Demo Kit

This kit demonstrates four core Python data libraries — **NumPy**, **Matplotlib**, **Seaborn**, and **Pandas** — using a connected retail/restaurant analytics scenario. It contains 8 files total: 4 runnable scripts and 4 companion explanation documents.

---

## 1. Files in This Kit

| # | File | Type | Purpose |
|---|---|---|---|
| 1 | `01_numpy_demo.py` | Executable script | NumPy arrays, stats, discounting, random data, correlation |
| 2 | `01_numpy_explanation.md` | Documentation | Line-by-line explanation of the NumPy script |
| 3 | `02_matplotlib_demo.py` | Executable script | 4-panel sales dashboard (line, bar, pie, scatter) |
| 4 | `02_matplotlib_explanation.md` | Documentation | Line-by-line explanation of the Matplotlib script |
| 5 | `03_seaborn_demo.py` | Executable script | Statistical dashboard on the `tips` dataset |
| 6 | `03_seaborn_explanation.md` | Documentation | Line-by-line explanation of the Seaborn script |
| 7 | `04_pandas_demo.py` | Executable script | DataFrames, filtering, grouping, missing data, CSV export |
| 8 | `04_pandas_explanation.md` | Documentation | Line-by-line explanation of the Pandas script |

**Suggested order:** run `04` → `01` → `02` → `03`, reading each `_explanation.md` alongside its script. In a real workflow, Pandas usually comes first (structuring and cleaning raw data), followed by NumPy (numeric computation) and then Matplotlib/Seaborn (visualization) — though each script also runs independently since all four use the same underlying sample data.

---

## 2. Prerequisites

- **Python 3.8 or later** installed on your machine.
- The following packages: `numpy`, `matplotlib`, `seaborn`, `pandas`.

Check your Python version:
```bash
python3 --version
```

Install the required packages (skip any already installed):
```bash
pip install numpy matplotlib seaborn pandas      # or: pip install -r requirements.txt
```
If you're on a system that separates Python environments strictly (e.g., some Linux distros), you may need:
```bash
pip install numpy matplotlib seaborn pandas --break-system-packages
```
or install inside a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install numpy matplotlib seaborn pandas      # or: pip install -r requirements.txt
```

---

## 3. How to Run Each File

Place all 8 files in the same folder, then open a terminal in that folder.

### Run the Pandas demo
```bash
python3 04_pandas_demo.py
```
**What happens:** prints a tidy DataFrame, filtering/sorting examples, grouped summaries, a computed "sales per visitor" column, and a missing-data cleanup demo — and **saves a CSV file**, `sales_summary.csv`, in the same folder. Open the CSV in Excel, Google Sheets, or a text editor to see the exported store-level summary.

### Run the NumPy demo
```bash
python3 01_numpy_demo.py
```
**What happens:** prints array creation, per-store/per-day statistics, a discounted sales table, simulated footfall data, and a correlation result — directly to the terminal. No files are created.

### Run the Matplotlib demo
```bash
python3 02_matplotlib_demo.py
```
**What happens:** prints a short summary to the terminal and **saves an image file**, `sales_dashboard.png`, in the same folder — a 4-panel chart (line, bar, pie, scatter). Open the PNG with any image viewer to see the dashboard.

### Run the Seaborn demo
```bash
python3 03_seaborn_demo.py
```
**Needs an internet connection the first time:** `sns.load_dataset("tips")` downloads the sample dataset from GitHub.

**What happens:** prints the loaded dataset preview, summary statistics, and key insights to the terminal, and **saves an image file**, `tips_dashboard.png` — a 4-panel statistical dashboard (histogram, regression plot, box plot, heatmap).

### Read the explanation files
The `.md` files are plain-text Markdown — open them in:
- Any text editor (Notepad, VS Code, etc.), or
- A Markdown viewer / GitHub / most note-taking apps, for nicely formatted headings and tables.

No installation or execution needed for these — they're reference documents, not code.

---

## 4. Expected Outputs After Running Everything

After running all four scripts once, your folder should contain:
```
01_numpy_demo.py
01_numpy_explanation.md
02_matplotlib_demo.py
02_matplotlib_explanation.md
03_seaborn_demo.py
03_seaborn_explanation.md
04_pandas_demo.py
04_pandas_explanation.md
sales_dashboard.png       <- created by running 02_matplotlib_demo.py
tips_dashboard.png        <- created by running 03_seaborn_demo.py
sales_summary.csv         <- created by running 04_pandas_demo.py
```

---

## 5. Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'numpy'` (or matplotlib/seaborn) | Package not installed | Run the `pip install` command in Section 2 |
| `command not found: python3` | Python not installed or not on PATH | Install Python 3 from python.org, or use `python` instead of `python3` depending on your system |
| Plot window doesn't pop up | `plt.show()` is intentionally commented out in these scripts (they save to PNG instead, for compatibility with non-interactive environments) | Open the saved `.png` file directly, or uncomment `plt.show()` near the bottom of the Matplotlib/Seaborn scripts if running locally with a display |
| `pip install` fails with a "externally managed environment" error | Your OS restricts system-wide pip installs | Use `pip install ... --break-system-packages` or create a virtual environment as shown in Section 2 |

---

## 6. Quick Reference: What Each Library Is For

- **NumPy** → fast numerical arrays and math (the foundation everything else builds on)
- **Pandas** → labeled tables (DataFrames): loading, filtering, grouping, cleaning data
- **Matplotlib** → general-purpose, fully customizable plotting
- **Seaborn** → statistical plotting on tabular/DataFrame data, built on top of Matplotlib

For the full breakdown of *why* each line of code is written the way it is, see the matching `_explanation.md` file for that library.
