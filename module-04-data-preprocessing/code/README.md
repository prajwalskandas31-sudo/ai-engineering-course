# Module 4: Practical Data Preprocessing & Feature Engineering Lab

Welcome to the **Module 4 Hands-On Practical Lab**! This project is designed for anyone with basic Python knowledge to understand, execute, and master the full **Data Preprocessing Pipeline** required before building machine learning models.

---

## 🎯 What You Will Learn & Implement

In this lab, you will transform a dirty, real-world customer dataset into a pristine, model-ready numerical matrix using Python (`pandas`, `numpy`, and `scikit-learn`).

You will execute and understand 8 core preprocessing techniques:

1. **Raw Data Diagnostics**: Inspecting data structures using `df.info()`, `df.describe()`, and `df.isnull().sum()`.
2. **String Parsing & Text Extraction**: Cleaning string currency values (e.g., `"$50,000"`) into floating-point numbers.
3. **Outlier Detection & Capping (IQR Rule)**: Mathematically bounding extreme values (e.g., Age = 120) using the Interquartile Range ($IQR = Q3 - Q1$).
4. **Missing Value Imputation**: Applying median imputation for continuous numeric features and mode imputation for categorical text features.
5. **Ordinal Encoding**: Converting ordered text categories (e.g., Education: High School < Bachelor < Master < PhD) into numerical ranks ($0, 1, 2, 3$).
6. **Nominal One-Hot Encoding**: Expanding unordered geographical text categories into binary indicator columns while dropping the first dummy variable to avoid the **Dummy Variable Trap** (perfect multicollinearity).
7. **Train-Test Splitting**: Partitioning features ($X$) and target ($y$) into 70% training and 30% testing subsets with reproducible shuffling (`random_state=42`).
8. **Feature Scaling & Preventing Data Leakage**: Standardizing numeric features ($Z = \frac{x - \mu}{\sigma}$) by fitting the scaler **strictly on the training set** and applying learned parameters to the test set.

---

## 🚀 How to Run the Code

### Prerequisites
Make sure you have Python 3 installed along with the essential data science libraries:
```bash
pip install pandas numpy scikit-learn
```

### Running the Lab
Simply execute the Python script from your terminal or command prompt:
```bash
pip install -r requirements.txt   # optional shortcut
python module4_practical_lab.py
```

---

## 🛠️ Code Breakdown & Architectural Workflow

### Step 1: Raw Data Ingestion
We create a synthetic dataset containing real-world flaws: missing values (`NaN`), currency formatting symbols (`$`, `,`), string categories, outliers, and target labels.

### Step 2: Diagnostic Checks
Using `df.info()` reveals data type mismatches where numerical columns were incorrectly loaded as strings.

### Step 3: Text Parsing
```python
df['Annual_Salary'] = df['Annual_Salary'].str.replace('$', '').str.replace(',', '')
df['Annual_Salary'] = pd.to_numeric(df['Annual_Salary'], errors='coerce')
```
Strips text symbols to enable mathematical computation.

### Step 4: IQR Outlier Capping
$$\text{Upper Limit} = Q3 + 1.5 \times IQR$$
Restricts extreme outlier values without deleting entire record rows.

### Step 5: Imputation
- **Median Imputation**: Used for `Age` and `Annual_Salary` because median is robust against skewed distributions and outliers.
- **Mode Imputation**: Used for `Education_Level` and `City` to fill missing values with the most frequently occurring label.

### Step 6: Encoding
- **Ordinal Mapping**: Preserves rank order for `Education_Level`.
- **One-Hot Encoding**: `pd.get_dummies(..., drop_first=True)` converts `City` into numerical binary flags.

### Step 7: Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
Ensures model performance is evaluated on unseen data to detect overfitting.

### Step 8: Standard Scaling without Data Leakage
```python
scaler = StandardScaler()
X_train_scaled['Age'] = scaler.fit_transform(X_train[['Age']])
X_test_scaled['Age'] = scaler.transform(X_test[['Age']])
```
- `fit_transform()` calculates $\mu$ and $\sigma$ **only** from `X_train`.
- `transform()` applies those exact training parameters to `X_test`, guaranteeing test set independence!
