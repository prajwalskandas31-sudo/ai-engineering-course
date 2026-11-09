# Module 6 Lab: Supervised Learning Algorithms in Python

This lab turns the Module 6 theory into working `scikit-learn` code: one regression model and six classifiers, all trained on the same customer dataset.

## Scenario: e-commerce customer analytics

You're an ML engineer at an e-commerce company. For 600 (synthetic) customers you have:

| Column | Meaning |
|---|---|
| `Age` | Years, 18–69 |
| `Monthly_Visits` | Site visits per month |
| `Support_Calls` | Customer-service tickets |
| `Discount_Score` | Sensitivity to promos, 1–10 |
| `Annual_Spend` | Total spend in $ (regression target) |
| `Churn` | 1 = customer left, 0 = stayed (classification target) |

Two tasks:

1. **Regression:** predict `Annual_Spend` with linear regression.
2. **Classification:** predict `Churn` with six algorithms: logistic regression, decision tree, random forest, SVM, KNN and naive Bayes.

## Theory → code

| Concept | In the code |
|---|---|
| Least squares / residuals | `LinearRegression().fit()` minimises the sum of squared residuals |
| Logistic S-curve, maximum likelihood | `LogisticRegression()`, `predict_proba()` gives the probability |
| Gini impurity and splitting | `DecisionTreeClassifier(max_depth=3)` |
| Bootstrapping and bagging | `RandomForestClassifier(n_estimators=100)` |
| Soft margin and the RBF kernel trick | `SVC(kernel='rbf', C=1.0)` |
| Majority vote of nearest neighbours | `KNeighborsClassifier(n_neighbors=5)` |
| Bayes' rule, independent features | `GaussianNB()` |

**Scaling:** SVM, KNN and logistic regression get standardised features (`StandardScaler`, fitted on the training set only). Trees and forests use the raw features, because splits like "Age > 30" don't change when you rescale.

## Run it

```bash
pip install -r requirements.txt
python supervised_learning_lesson.py
```

## Expected results

### Part A: linear regression

| | Learned | True value used to generate the data |
|---|---|---|
| Intercept | $206.91 | $200 |
| Age | +$12.33 per year | +$12 |
| Monthly visits | +$50.81 per visit | +$50 |
| Support calls | −$28.41 per call | −$25 |
| Discount score | −$4.07 | 0 (not used) |

**R² = 0.737**: the model explains about 74% of the variation in annual spend. RMSE is about $144, close to the $150 of random noise built into the data, so the model has found nearly all the signal there is.

### Part B: six classifiers

About 1 in 5 test customers churn, so a "model" that always predicts *retained* already scores **78% accuracy** while catching **no** churners. That's why the lab also reports **churn recall** (the share of real churners caught) and **F1**.

| Algorithm | Accuracy | Churn recall | Churn F1 |
|---|---|---|---|
| Baseline (always "retained") | 78.0% | 0% | – |
| Logistic regression | 96.7% | 87.9% | 0.92 |
| Decision tree (depth 3) | 89.3% | 69.7% | 0.74 |
| Random forest (100 trees) | 94.7% | 78.8% | 0.87 |
| **SVM (RBF)** | **97.3%** | **87.9%** | **0.94** |
| KNN (K = 5) | 94.7% | 81.8% | 0.87 |
| Naive Bayes | 94.0% | 84.8% | 0.86 |

Why the scores are so high: churn in this synthetic data is a fixed rule of the features with no randomness, and the rule is roughly linear in support calls, visits, spend and discount score. That suits logistic regression and the SVM. Real churn data is much noisier, so expect lower scores there.

### Step 3: one high-risk customer

A customer with 6 support calls, 2 visits a month and $120 spend: both logistic regression and the random forest predict **churn**, with probabilities near 100% and 98%.

## Exercises

1. **Tune K:** change `n_neighbors` from 5 to 1 and to 25. K = 1 follows every noisy point; K = 25 smooths the boundary. Watch accuracy and recall.
2. **Feature importance:** print `classifiers["3. Random Forest (100 trees)"].feature_importances_` to see which behaviours drive churn.
3. **Deeper tree:** try `max_depth=None` and compare training vs test accuracy to see a single tree overfit.

## Changes from the original script

- Comments now match the code (the spend formula uses 12 × Age and −25 × calls; the sample customer has 6 calls and $120 spend).
- Added the majority-class baseline, churn recall and F1, because accuracy alone is misleading with imbalanced classes.
- Removed `probability=True` from `SVC`: it wasn't used and is deprecated in recent scikit-learn.
