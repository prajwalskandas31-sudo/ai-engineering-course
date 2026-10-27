"""
================================================================================
 MODULE 3: MATHEMATICS FOR AI -- WORKED EXAMPLES IN PYTHON
================================================================================
Every worked example and practice answer from the Module 3 notes, recomputed
with Python and NumPy so you can check the maths yourself and change the
numbers to experiment.

Sections:
    1. Mean, median, mode (and why the median beats the mean with outliers)
    2. Standard deviation (population vs sample) and a simple fraud check
    3. Probability
    4. Vectors and matrices (matrix-vector multiplication)
    5. Derivatives with the power rule
    6. Gradient descent: one update step, then a full loop

Run:  python3 math_for_ai_examples.py      (needs: pip install numpy)
================================================================================
"""

import statistics
import numpy as np


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------------
header("1. MEAN, MEDIAN, MODE")
# ----------------------------------------------------------------------------
marks = [10, 20, 20, 30, 40]
print(f"Marks: {marks}")
print(f"  Mean   = {np.mean(marks)}")              # 24
print(f"  Median = {np.median(marks)}")            # 20
print(f"  Mode   = {statistics.mode(marks)}")      # 20

# Outliers: 4 junior developers + 1 CEO
salaries = [30_000, 30_000, 30_000, 30_000, 450_000]
print(f"\nSalaries: {salaries}")
print(f"  Mean   = Rs {np.mean(salaries):,.0f}   <- pulled up by one outlier")
print(f"  Median = Rs {np.median(salaries):,.0f}    <- what a typical person earns")

# Google Maps style travel times with one traffic spike
travel = [15, 16, 15, 17, 60]
print(f"\nTravel times (mins): {travel}")
print(f"  Mean = {np.mean(travel)} mins | Median = {np.median(travel)} mins")

# Practice problem 1
practice = [5, 7, 8, 8, 9, 12]
print(f"\nPractice 1 {practice}: mean = {np.mean(practice):.2f}, "
      f"median = {np.median(practice)}, mode = {statistics.mode(practice)}")

# ----------------------------------------------------------------------------
header("2. STANDARD DEVIATION")
# ----------------------------------------------------------------------------
data = [1, 3, 5, 7, 9]
print(f"Data: {data}")
print(f"  Mean                  = {np.mean(data)}")
print(f"  Population variance   = {np.var(data)}          (divide by N)")
print(f"  Population SD (sigma) = {np.std(data):.2f}")
print(f"  Sample SD (s)         = {np.std(data, ddof=1):.2f}       (divide by n - 1)")

# A simple fraud check: flag anything more than 3 SDs from the usual spend
daily_spend = np.array([520, 780, 640, 910, 700, 560, 850, 990, 610, 730])
new_txn = 200_000
mean, sd = daily_spend.mean(), daily_spend.std()
z = (new_txn - mean) / sd
print(f"\nUsual daily spend: mean Rs {mean:,.0f}, SD Rs {sd:,.0f}")
print(f"New transaction Rs {new_txn:,} is {z:,.0f} SDs above normal "
      f"-> {'FRAUD ALERT' if z > 3 else 'looks normal'}")

# ----------------------------------------------------------------------------
header("3. PROBABILITY")
# ----------------------------------------------------------------------------
red, blue, green = 5, 3, 2
p_blue = blue / (red + blue + green)
print(f"P(blue ball) = {blue} / {red + blue + green} = {p_blue:.2f} ({p_blue:.0%})")

p_spam = 0.98
print(f"Spam filter: P(spam) = {p_spam} > 0.5 -> "
      f"{'Spam folder' if p_spam > 0.5 else 'Inbox'}")

# ----------------------------------------------------------------------------
header("4. VECTORS AND MATRICES")
# ----------------------------------------------------------------------------
A = np.array([[1, 2],
              [3, 4]])
v = np.array([[2],
              [1]])
print("A =\n", A)
print("v =\n", v)
print("A @ v =\n", A @ v)                        # [[4], [10]]

M = np.zeros((5, 6))
print(f"\nA 5 x 6 matrix has shape {M.shape} and {M.size} elements")

# ----------------------------------------------------------------------------
header("5. DERIVATIVES (power rule: d/dx x^n = n * x^(n-1))")
# ----------------------------------------------------------------------------
def numerical_derivative(f, x, h=1e-6):
    """Slope of f at x, estimated from two very close points."""
    return (f(x + h) - f(x - h)) / (2 * h)

x = 2.0
for label, f, exact in [
    ("x^2   -> 2x  ", lambda t: t ** 2,     lambda t: 2 * t),
    ("3x^2  -> 6x  ", lambda t: 3 * t ** 2, lambda t: 6 * t),
    ("5x^3  -> 15x^2", lambda t: 5 * t ** 3, lambda t: 15 * t ** 2),
]:
    print(f"  {label}  at x = {x}: rule gives {exact(x):.2f}, "
          f"numerical slope = {numerical_derivative(f, x):.2f}")

# ----------------------------------------------------------------------------
header("6. GRADIENT DESCENT")
# ----------------------------------------------------------------------------
def update(theta, alpha, gradient):
    """theta_new = theta_old - alpha * gradient"""
    return theta - alpha * gradient

print(f"Worked example: theta = 5, alpha = 0.1, gradient = 8 -> "
      f"{update(5, 0.1, 8):.1f}")
print(f"Practice:       theta = 10, alpha = 0.05, gradient = 6 -> "
      f"{update(10, 0.05, 6):.1f}")

# Full loop: minimise the loss J(theta) = (theta - 3)^2, whose lowest point is theta = 3
def loss(theta):
    return (theta - 3) ** 2

def gradient(theta):
    return 2 * (theta - 3)          # derivative of (theta - 3)^2

for alpha in (0.1, 1.1):
    theta = 10.0
    print(f"\nLearning rate alpha = {alpha}, starting at theta = {theta}")
    for step in range(1, 6):
        theta = update(theta, alpha, gradient(theta))
        print(f"  step {step}: theta = {theta:8.3f}   loss = {loss(theta):10.3f}")
    verdict = "converging on 3" if abs(theta - 3) < 3 else "overshooting: step too big!"
    print(f"  -> {verdict}")

print("\nDone. Change the numbers above and re-run to explore.")
