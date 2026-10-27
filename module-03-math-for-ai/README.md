# Module 3: Mathematics for AI

← [Module 2](../module-02-python-for-ai) · [Course home](../README.md) · 📒 [Notes on Notion](https://chipped-willow-4a2.notion.site/4066f1ed006e814782dddd9efb5bdaeb) · ▶️ [Module 3 playlist](https://www.youtube.com/playlist?list=PLA_tEeKDxyfU) · Next: Module 4 (coming soon)

## What you'll learn

- Descriptive vs inferential statistics
- Mean, median and mode, and why the median wins when there are outliers
- Standard deviation (population vs sample) and how it powers fraud detection
- Probability as a confidence score (spam filters, medical AI)
- Vectors and matrices: how images and words become numbers
- Derivatives with the power rule
- Gradient descent: how AI learns from its mistakes

## Video

| # | Video | Type | Length |
|---|---|---|---|
| 1 | [The Mathematics of Mistakes](https://youtu.be/Ceo1jMAIVUk) | Explainer | 4:35 |

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

> **Correction:** in the salary example (4 × ₹30,000 + 1 × ₹4,50,000), the mean is **₹1,14,000**. The visual deck rounds this to ₹1.13L; the correct figure is ₹1.14L.
