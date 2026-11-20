# Module 7 Lab: Customer Segmentation & Outlier Detection

This lab applies all four Module 7 algorithms to one unlabelled dataset: **PCA**, **K-means**, **hierarchical clustering** and **DBSCAN**.

## Scenario

An e-commerce store has 300 customers and no labels. Nobody has told us who is a big spender, who is a bargain hunter, or which accounts look fraudulent. For each customer we have 5 features:

| Feature | Meaning |
|---|---|
| `Annual_Income_k` | Income in $1,000s |
| `Spending_Score` | 1–100 |
| `Purchase_Freq` | Orders per year |
| `Return_Rate_Pct` | % of products returned |
| `Session_Duration_Min` | Average minutes per visit |

Hidden inside the data (the algorithms are never told): **three personas** (100 budget shoppers, 100 big spenders, 90 high-income low spenders) and **10 planted anomalous accounts** (rows 290–299).

## Theory → code

| Algorithm | Idea | In the code |
|---|---|---|
| Scaling | Distances need comparable units | `StandardScaler()` |
| PCA | Keep the directions with the most variation | `PCA(n_components=2)`, `explained_variance_ratio_`, `components_` (loadings) |
| K-means | Assign to nearest centre, move centre, repeat | `KMeans(n_clusters=k, n_init=10)`, `inertia_` for the elbow plot |
| Hierarchical | Merge the closest clusters bottom-up | `linkage(X, method='ward')`, `dendrogram(...)`, `AgglomerativeClustering` |
| DBSCAN | Grow clusters through dense regions, flag the rest as noise | `DBSCAN(eps=0.8, min_samples=5)`; label `-1` = noise |

## Run it

```bash
pip install -r requirements.txt
python customer_segmentation_lesson.py
```

It prints each step and saves `unsupervised_learning_master_plot.png` (elbow plot, K-means clusters, dendrogram, DBSCAN outliers) in the folder you run it from.

## Expected results

**PCA:** PC1 explains 60.6% and PC2 20.1%, so the 2D view keeps **80.7%** of the variation.

| Feature | PC1 loading | PC2 loading |
|---|---|---|
| Income | 0.42 | 0.22 |
| Spending | 0.51 | −0.25 |
| Frequency | 0.53 | −0.17 |
| Return rate | 0.15 | **0.93** |
| Duration | 0.51 | −0.03 |

PC1 is "overall activity" (spending, frequency, session time); PC2 is almost entirely return rate.

**K-means elbow:** inertia goes 1500 → 756 → 515 → 303 → 201 → 116 for K = 1–6. The bend is gradual, not sharp: the 10 outliers keep tempting K-means into extra tiny clusters. K = 3 is chosen because it matches the three customer types the business expects.

**K-means personas (K = 3):**

| Persona | Income ($k) | Spending | Orders/yr | Returns % | Session (min) | Customers |
|---|---|---|---|---|---|---|
| Big spenders | 86 | 79 | 47 | 9 | 39 | 105 |
| High-income, low spend | 88 | 26 | 15 | 14 | 19 | 95 |
| Budget shoppers | 30 | 20 | 10 | 5 | 12 | 100 |

The big spender and high-income groups each absorbed 5 of the outliers. K-means has no "noise" option.

**Hierarchical (Ward):** agrees with K-means almost exactly (Adjusted Rand Index **0.98**).

**DBSCAN (ε = 0.8, MinPts = 5):** finds the **3 personas** and flags **exactly the 10 planted anomalies** as noise.

## Exercises

1. **Scaling matters:** run K-means on the raw `df` instead of `X_scaled`. Which feature takes over?
2. **DBSCAN sensitivity:** try `eps=1.2`: the budget shoppers and high-income low spenders merge into one cluster and 2 anomalies slip through. At `eps=2.5` everything becomes one cluster.
3. **Read the loadings:** which features would you look at to explain PC1? PC2?
4. **Cut the tree:** change `n_clusters` in `AgglomerativeClustering` to 2 and 4. Where do the merges happen on the dendrogram?

## Changes from the original script

- **Plot path:** the original saved the plot to `/workspace/scratch/...`, which doesn't exist on most machines and crashes the script. It now saves next to where you run it.
- **DBSCAN epsilon 1.2 → 0.8:** at 1.2, DBSCAN merged two of the three personas and caught only 8 of the 10 anomalies. At 0.8 it finds all three personas and all ten anomalies.
- **New step 5:** prints persona profiles, K-means vs hierarchical agreement, and which customers DBSCAN flagged.
- **Elbow comment:** notes that the elbow on this data is soft and K = 3 comes from domain knowledge.
