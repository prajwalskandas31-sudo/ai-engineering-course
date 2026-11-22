# Module 3: Mathematics for AI

← [Module 2](../module-02-python-for-ai) · [Course home](../README.md) · 📒 [Notes on Notion](https://chipped-willow-4a2.notion.site/4066f1ed006e814782dddd9efb5bdaeb) · ▶️ [Module 3 playlist](https://www.youtube.com/playlist?list=PLA_tEeKDxyfU) · Next: [Module 4](../module-04-data-preprocessing)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prajwalskandas31-sudo/ai-engineering-course/blob/main/module-03-math-for-ai/Module_3_Lab.ipynb) **Run this module's lab in your browser** with Google Colab: no installs needed. Notebook: [Module_3_Lab.ipynb](Module_3_Lab.ipynb) · 📄 Step-by-step [code walkthrough with real outputs (PDF)](Module_3_Code_Walkthrough.pdf)

## What you'll learn

- Descriptive vs inferential statistics
- Mean, median and mode, and why the median wins when there are outliers
- Standard deviation (population vs sample) and how it powers fraud detection
- Probability as a confidence score (spam filters, medical AI)
- Vectors and matrices: how images and words become numbers
- Derivatives with the power rule
- Gradient descent: how AI learns from its mistakes

## Videos

| # | Video | Type | Length |
|---|---|---|---|
| 1 | [The Mathematics of Mistakes](https://youtu.be/Ceo1jMAIVUk) | Explainer | 4:35 |
| 2 | [The Maths Behind AI, in Python (lab walkthrough)](https://youtu.be/gZUAf8GmgVU) | Lab Walkthrough | 5:26 |

## Run the worked examples

[code/math_for_ai_examples.py](code/math_for_ai_examples.py) recomputes every worked example and practice answer from the notes, and shows gradient descent converging (α = 0.1) and overshooting (α = 1.1).

```bash
cd code
pip install -r requirements.txt
python3 math_for_ai_examples.py
```

## Files

| File | Description |
|---|---|
| [slides/Module_3_Mathematics_for_AI.pptx](slides/Module_3_Mathematics_for_AI.pptx) | Lecture slides |
| [slides/AI_Math_Foundations_Visual_Deck.pdf](slides/AI_Math_Foundations_Visual_Deck.pdf) | Visual study deck |
| [notes/Module_3_Student_Notes.pdf](notes/Module_3_Student_Notes.pdf) | Student study and practice guide |
| [code/math_for_ai_examples.py](code/math_for_ai_examples.py) | All worked examples in Python + NumPy |
| [slides/Mathematics_for_AI_Infographic.png](slides/Mathematics_for_AI_Infographic.png) | One-page infographic: the module at a glance |

> **Correction:** in the salary example (4 × ₹30,000 + 1 × ₹4,50,000), the mean is **₹1,14,000**. The visual deck rounds this to ₹1.13L; the correct figure is ₹1.14L.
