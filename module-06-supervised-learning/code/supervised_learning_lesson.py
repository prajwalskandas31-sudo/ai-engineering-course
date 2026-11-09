"""
===============================================================================
MODULE 6: SUPERVISED LEARNING ALGORITHMS - PRACTICAL LAB LESSON
===============================================================================
A complete, beginner-friendly hands-on implementation connecting ML theory to 
clean Python code using Scikit-Learn.

Dataset Scenario: E-Commerce Customer Analytics
- Task 1 (Regression): Predict Customer Annual Spend ($) using Linear Regression
- Task 2 (Classification): Predict Customer Churn (1 = Churned, 0 = Retained) 
  using 6 Supervised Classification Algorithms:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. Random Forest Classifier
  4. Support Vector Machine (SVM)
  5. K-Nearest Neighbors (KNN)
  6. Naive Bayes (GaussianNB)
===============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, recall_score, f1_score

# Machine Learning Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Set random seed for reproducible results
np.random.seed(42)

def generate_synthetic_customer_data(n_samples=500):
    """
    Generates a realistic synthetic e-commerce customer dataset.
    Features:
    - Age (years)
    - Monthly Visits (count)
    - Customer Support Calls (count)
    - Discount Usage Score (1-10)
    - Annual Spend ($) [Continuous Target for Regression]
    - Churn (0 or 1) [Discrete Target for Classification]
    """
    age = np.random.randint(18, 70, size=n_samples)
    monthly_visits = np.random.poisson(lam=8, size=n_samples) + 1
    support_calls = np.random.poisson(lam=2, size=n_samples)
    discount_score = np.random.uniform(1, 10, size=n_samples)
    
    # Target 1: Annual Spend ($) = 200 + 50*Visits + 12*Age - 25*SupportCalls + Noise
    annual_spend = 200 + (50 * monthly_visits) + (12 * age) - (25 * support_calls) + np.random.normal(0, 150, n_samples)
    annual_spend = np.maximum(annual_spend, 50) # Floor spend at $50
    
    # Target 2: Churn driven by high support calls, low visits, low spend.
    # Note: churn is a fixed rule of the features (no random noise), so the
    # classifiers can score very high. About 17% of customers churn.
    churn_score = 0.4 * support_calls - 0.15 * monthly_visits - 0.001 * annual_spend + 0.1 * discount_score
    churn_prob = 1 / (1 + np.exp(-churn_score)) # Logistic sigmoid transformation
    churn = (churn_prob > 0.45).astype(int)
    
    df = pd.DataFrame({
        'Age': age,
        'Monthly_Visits': monthly_visits,
        'Support_Calls': support_calls,
        'Discount_Score': np.round(discount_score, 1),
        'Annual_Spend': np.round(annual_spend, 2),
        'Churn': churn
    })
    return df

def run_practical_lesson():
    print("==================================================================")
    print("  STEP 1: DATA EXPLORATION & PREPARATION")
    print("==================================================================")
    df = generate_synthetic_customer_data(n_samples=600)
    print(f"Generated dataset with {len(df)} customer records.\n")
    print("First 5 records:")
    print(df.head())
    print("\nDataset Statistics:")
    print(df.describe().T[['mean', 'std', 'min', '50%', 'max']])
    
    # -------------------------------------------------------------------------
    # PART A: REGRESSION (Linear Regression)
    # -------------------------------------------------------------------------
    print("\n==================================================================")
    print("  PART A: LINEAR REGRESSION (Predicting Annual Spend $)")
    print("==================================================================")
    X_reg = df[['Age', 'Monthly_Visits', 'Support_Calls', 'Discount_Score']]
    y_reg = df['Annual_Spend']
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_r, y_train_r)
    y_pred_r = lin_reg.predict(X_test_r)
    
    r2 = r2_score(y_test_r, y_pred_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    
    print(f"Linear Regression Intercept (b): ${lin_reg.intercept_:.2f}")
    print("Learned Feature Coefficients (Slopes):")
    for col, coef in zip(X_reg.columns, lin_reg.coef_):
        print(f"  - {col:15s}: {coef:+.2f} $/unit")
    print(f"\nModel Performance Metrics:")
    print(f"  - Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"  - R-squared (R2 Score):           {r2:.4f} ({r2*100:.1f}% variance explained)")

    # -------------------------------------------------------------------------
    # PART B: CLASSIFICATION (Comparing 6 Supervised Algorithms)
    # -------------------------------------------------------------------------
    print("\n==================================================================")
    print("  PART B: CLASSIFICATION (Predicting Customer Churn 0 vs 1)")
    print("==================================================================")
    X_cls = df[['Age', 'Monthly_Visits', 'Support_Calls', 'Discount_Score', 'Annual_Spend']]
    y_cls = df['Churn']
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.25, random_state=42)
    
    # Feature Scaling for Distance-based models (KNN, SVM, Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_c)
    X_test_scaled = scaler.transform(X_test_c)
    
    classifiers = {
        "1. Logistic Regression": LogisticRegression(),
        "2. Decision Tree (Depth=3)": DecisionTreeClassifier(max_depth=3, random_state=42),
        "3. Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, random_state=42),
        "4. SVM (RBF Kernel)": SVC(kernel='rbf', C=1.0, random_state=42),
        "5. K-Nearest Neighbors (K=5)": KNeighborsClassifier(n_neighbors=5),
        "6. Naive Bayes (Gaussian)": GaussianNB()
    }
    
    results = []
    
    # Baseline: always predict the majority class ("retained")
    baseline = 1 - y_test_c.mean()
    print(f"Churn rate in test set: {y_test_c.mean()*100:.1f}%")
    print(f"Baseline (always predict 'retained'): {baseline*100:.2f}% accuracy, 0% of churners caught\n")

    print(f"{'Algorithm':<30} | {'Accuracy':>8} | {'Churn recall':>12} | {'Churn F1':>8} | Theoretical connection")
    print("-" * 110)
    
    for name, model in classifiers.items():
        # Scale features for distance/gradient models, use unscaled for tree models
        if "Decision Tree" in name or "Random Forest" in name:
            model.fit(X_train_c, y_train_c)
            preds = model.predict(X_test_c)
        else:
            model.fit(X_train_scaled, y_train_c)
            preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test_c, preds)
        rec = recall_score(y_test_c, preds)
        f1 = f1_score(y_test_c, preds)
            
        results.append((name, acc))
        
        # Mapping theoretical note
        notes = {
            "1. Logistic Regression": "Fits S-curve via Maximum Likelihood Estimation",
            "2. Decision Tree (Depth=3)": "Splits on features minimizing Gini Impurity",
            "3. Random Forest (100 trees)": "Ensemble of decision trees with Bootstrap Bagging",
            "4. SVM (RBF Kernel)": "Maximizes Soft Margin using Infinite-Dim Kernel Trick",
            "5. K-Nearest Neighbors (K=5)": "Votes based on Euclidean distance to 5 closest points",
            "6. Naive Bayes (Gaussian)": "Applies Bayes Rule assuming feature independence"
        }
        print(f"{name:<30} | {acc*100:7.2f}% | {rec*100:11.1f}% | {f1:8.2f} | {notes[name]}")

    print("\nTip: with only about 1 in 5 customers churning, accuracy alone flatters every model.")
    print("Churn recall = the share of real churners each model actually catches.")

    print("\n==================================================================")
    print("  STEP 3: SAMPLE INDIVIDUAL PREDICTION TEST")
    print("==================================================================")
    # Test a high-risk customer profile: 6 Support Calls, 2 Monthly Visits, $120 Annual Spend
    sample_customer = pd.DataFrame([{
        'Age': 45,
        'Monthly_Visits': 2,
        'Support_Calls': 6,
        'Discount_Score': 8.5,
        'Annual_Spend': 120.0
    }])
    
    sample_scaled = scaler.transform(sample_customer)
    
    rf_model = classifiers["3. Random Forest (100 trees)"]
    lr_model = classifiers["1. Logistic Regression"]
    
    rf_pred = rf_model.predict(sample_customer)[0]
    rf_prob = rf_model.predict_proba(sample_customer)[0][1]
    
    lr_pred = lr_model.predict(sample_scaled)[0]
    lr_prob = lr_model.predict_proba(sample_scaled)[0][1]
    
    print("Profile: High support calls (6), low visits (2), low spend ($120)")
    print(f" -> Logistic Regression Prediction : Churn={lr_pred} (Probability: {lr_prob*100:.1f}%)")
    print(f" -> Random Forest Prediction       : Churn={rf_pred} (Probability: {rf_prob*100:.1f}%)")
    print("==================================================================")

if __name__ == "__main__":
    run_practical_lesson()
