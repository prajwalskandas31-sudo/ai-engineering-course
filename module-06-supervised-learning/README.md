# Module 6: Supervised Learning Algorithms

← [Module 5](../module-05-ml-fundamentals) · [Course home](../README.md) · 📒 [Notes on Notion](https://chipped-willow-4a2.notion.site/4136f1ed006e81339b2ff4a1fda3bd46) · ▶️ [Module 6 playlist](https://www.youtube.com/playlist?list=PLIX9fLOWQUKg) · Next: Module 7 (coming soon)

## What you'll learn

- Linear regression: least squares, residuals and R²
- Logistic regression: the S-curve and maximum likelihood
- Decision trees and Gini impurity, and why single trees overfit
- Random forests: bootstrapping, random features, voting and out-of-bag error
- K-nearest neighbours and choosing K
- Naive Bayes, the bag-of-words assumption and Laplace smoothing
- Support vector machines: maximal and soft margins, support vectors and the kernel trick

## Videos

| # | Video | Type | Length |
|---|---|---|---|
| 1 | [The Machine Learning Toolbox](https://youtu.be/xtW0Rw6LLXU) | Explainer | 8:26 |
| 2 | [Supervised ML Toolkit](https://youtu.be/6bA5q103pwU) | Explainer | 8:42 |
| 3 | [The ML Toolkit, Explained](https://youtu.be/3zEghngrVpk) | Explainer | 9:58 |
| 4 | [Supervised Learning Algorithms: Live Class](https://youtu.be/kWGx1UFLWYg) | Class Recording | 1:17:46 |

## Hands-on lab

[code/supervised_learning_lesson.py](code/supervised_learning_lesson.py) uses 600 synthetic e-commerce customers to predict **annual spend** with linear regression (R² = 0.74) and **churn** with six classifiers: logistic regression, decision tree, random forest, SVM, KNN and naive Bayes. See [code/README.md](code/README.md) for the walkthrough and expected results.

```bash
cd code
pip install -r requirements.txt
python supervised_learning_lesson.py
```

| Algorithm | Accuracy | Churn recall |
|---|---|---|
| Baseline (always "retained") | 78.0% | 0% |
| Logistic regression | 96.7% | 87.9% |
| Decision tree (depth 3) | 89.3% | 69.7% |
| Random forest | 94.7% | 78.8% |
| **SVM (RBF)** | **97.3%** | **87.9%** |
| KNN (K = 5) | 94.7% | 81.8% |
| Naive Bayes | 94.0% | 84.8% |

## Files

| File | Description |
|---|---|
| [code/supervised_learning_lesson.py](code/supervised_learning_lesson.py) | Regression + six classifiers lab |
| [code/README.md](code/README.md) | Lab guide |
| [slides/The_Machine_Learning_Blueprint.pdf](slides/The_Machine_Learning_Blueprint.pdf) | Visual deck: The Machine Learning Field Guide |
| [slides/The_Predictive_Blueprint.pdf](slides/The_Predictive_Blueprint.pdf) | Visual deck: Structural Engineering for Data |

> **Note on the lab:** the original script reported accuracy only. With about 1 in 5 customers churning, a model that always predicts "retained" already scores 78%, so this version also prints a baseline, churn recall and F1. A few comments were corrected to match the code. See the lab guide for details.
