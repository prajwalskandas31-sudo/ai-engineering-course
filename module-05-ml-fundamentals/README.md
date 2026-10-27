# Module 5: Machine Learning Fundamentals

← [Module 4](../module-04-data-preprocessing) · [Course home](../README.md) · 📒 [Notes on Notion](https://chipped-willow-4a2.notion.site/4066f1ed006e811d9770eb5c592b078f) · ▶️ [Module 5 playlist](https://www.youtube.com/playlist?list=PLL8oIdsEel2Y) · Next: Module 6 (coming soon)

## What you'll learn

- What machine learning is: classification vs prediction, supervised vs reinforcement learning
- Training data vs testing data, and measuring error with the sum of squares
- Bias, variance, underfitting and overfitting, and finding the sweet spot
- K-fold, 10-fold and leave-one-out cross-validation
- Reinforcement learning: agent, environment, policy, reward and the learning rate

## Videos

| # | Video | Type | Length |
|---|---|---|---|
| 1 | [Machine Learning Basics](https://youtu.be/CS7ADiZjSGo) | Explainer | 7:10 |
| 2 | [How Machines Learn to Make Predictions](https://youtu.be/9-ejhYU7HrU) | Explainer | 8:23 |
| 3 | [Machine Learning Fundamentals: Live Class](https://youtu.be/TNq3f9QHvx8) | Class Recording | 1:26:47 |

## Hands-on lab

[code/ml_practical_lesson.py](code/ml_practical_lesson.py) predicts a mouse's top running speed from its weight. It compares a straight line (degree 1), a curve (degree 2) and a squiggle (degree 10), and lets 5-fold cross-validation choose the winner before the test set is touched. See [code/README.md](code/README.md) for the walkthrough and expected results.

```bash
cd code
pip install -r requirements.txt
python ml_practical_lesson.py
```

![Underfit vs sweet spot vs overfit](outputs/bias_variance_fits.png)

## Files

| File | Description |
|---|---|
| [code/ml_practical_lesson.py](code/ml_practical_lesson.py) | Bias, variance and cross-validation lab |
| [code/README.md](code/README.md) | Lab guide |
| [notes/Module_5_ML_Fundamentals_Study_Guide.pdf](notes/Module_5_ML_Fundamentals_Study_Guide.pdf) | Study guide |
| [slides/Machine_Learning_Essentials.pdf](slides/Machine_Learning_Essentials.pdf) | Visual deck: ML essentials |
| [slides/Machine_Learning_Evaluation_Blueprint.pdf](slides/Machine_Learning_Evaluation_Blueprint.pdf) | Visual deck: the evaluation blueprint |

> **Note on the lab:** the original script printed fixed conclusions ("error explodes on test data" for degree 10). In this version the conclusions are computed from the actual numbers, and the features are scaled before the polynomial expansion so degree 10 fits properly. On this small 10-sample test split, degree 10 scores about the same as degree 2 on the test set, which is why model choice is made with cross-validation.
