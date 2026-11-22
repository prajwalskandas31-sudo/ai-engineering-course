# Module 4: Data Preprocessing

← [Module 3](../module-03-math-for-ai) · [Course home](../README.md) · 📒 [Notes on Notion](https://chipped-willow-4a2.notion.site/4066f1ed006e813891daf5bf13678b93) · ▶️ [Module 4 playlist](https://www.youtube.com/playlist?list=PLalpeLmg2joU) · Next: [Module 5](../module-05-ml-fundamentals)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prajwalskandas31-sudo/ai-engineering-course/blob/main/module-04-data-preprocessing/Module_4_Lab.ipynb) **Run this module's lab in your browser** with Google Colab: no installs needed. Notebook: [Module_4_Lab.ipynb](Module_4_Lab.ipynb) · 📄 Step-by-step [code walkthrough with real outputs (PDF)](Module_4_Code_Walkthrough.pdf)

## What you'll learn

- Inspecting raw data with `info()`, `describe()` and `isnull()`, and the data-type trap
- Exploratory data analysis: distributions, correlation, multicollinearity
- Removing exact and near-duplicates
- Handling missing values: when to drop, and median / mode / "Unknown" / forward-fill / KNN imputation
- Cleaning text and capping outliers with the IQR rule
- One-hot vs ordinal encoding, and the dummy variable trap
- Train-test split, standardization vs normalization, and preventing data leakage

## Videos

| # | Video | Type | Length |
|---|---|---|---|
| 1 | [Data Preprocessing Explained](https://youtu.be/OY0-7MTimSs) | Explainer | 7:38 |
| 2 | [Data Preprocessing in Python](https://youtu.be/Tw5s88F49Ho) | Explainer | 7:22 |
| 3 | [Cleaning Messy Data for Machine Learning (lab walkthrough)](https://youtu.be/5-h73M0xLa4) | Lab Walkthrough | 4:28 |

## Hands-on lab

[code/module4_practical_lab.py](code/module4_practical_lab.py) takes a messy 10-customer dataset (missing values, `"$50,000"` salaries, an age of 120) through an 8-step pipeline to a model-ready matrix. See [code/README.md](code/README.md) for the step-by-step guide.

```bash
cd code
pip install -r requirements.txt
python module4_practical_lab.py
```

## Files

| File | Description |
|---|---|
| [code/module4_practical_lab.py](code/module4_practical_lab.py) | 8-step preprocessing pipeline |
| [code/README.md](code/README.md) | Lab guide |
| [notes/Module_4_Preprocessing_Notes.pdf](notes/Module_4_Preprocessing_Notes.pdf) | Full 24-page study guide |
| [slides/The_Data_Refinery_Visual_Deck.pdf](slides/The_Data_Refinery_Visual_Deck.pdf) | Visual study deck |
| [slides/Data_Preprocessing_Infographic.png](slides/Data_Preprocessing_Infographic.png) | One-page infographic: the module at a glance |

> **Notes on the PDF:** (1) Standardization is described as "highly robust to outliers". It is less fragile than min-max scaling, but the mean and SD are still pulled by outliers; use `RobustScaler` or cap outliers first. (2) In pandas 3, `df['col'].fillna(..., inplace=True)` no longer updates `df` and `fillna(method='ffill')` raises an error. Use `df['col'] = df['col'].fillna(...)` and `.ffill()` instead. The Notion notes use the updated code.
