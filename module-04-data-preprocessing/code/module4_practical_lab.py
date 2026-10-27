import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_data_preprocessing_pipeline():
    print("=" * 70)
    print(" MODULE 4: PRACTICAL DATA PREPROCESSING PIPELINE (HANDS-ON LAB) ")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # STEP 1: CREATE A REAL-WORLD MESSY DATASET (SIMULATED RAW DATA)
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Generating Raw Messy Dataset...")
    raw_data = {
        'Customer_ID': ['C101', 'C102', 'C103', 'C104', 'C105', 'C106', 'C107', 'C108', 'C109', 'C110'],
        'Age': [25.0, 30.0, np.nan, 45.0, 120.0, 35.0, np.nan, 50.0, 28.0, 40.0],  # Missing values + Outlier (120)
        'Annual_Salary': ['$50,000', '$65,000', '$120,000', np.nan, '$85,000', '$95,000', '$45,000', '$110,000', np.nan, '$70,000'], # String with symbols & NaNs
        'Education_Level': ['Bachelor', 'Master', 'High School', 'PhD', 'Bachelor', np.nan, 'Master', 'PhD', 'Bachelor', 'Master'], # Ordinal + NaN
        'City': ['Mumbai', 'Delhi', 'Bangalore', 'Mumbai', np.nan, 'Delhi', 'Bangalore', 'Mumbai', 'Delhi', 'Bangalore'], # Nominal + NaN
        'Purchased': ['No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes'] # Target
    }
    
    df = pd.DataFrame(raw_data)
    print("\n--- Raw Uncleaned DataFrame ---")
    print(df)
    
    # -------------------------------------------------------------------------
    # STEP 2: DATA INSPECTION & DIAGNOSTICS
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[STEP 2] Diagnostic Check: Identifying Messy Structural Issues")
    print("-" * 70)
    print("\nData Types & Non-Null Counts:")
    print(df.info())
    
    print("\nMissing Values Count per Column:")
    print(df.isnull().sum())
    
    # -------------------------------------------------------------------------
    # STEP 3: TEXT PARSING & CLEANING (STRING TRANSFORMATION)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[STEP 3] Cleaning String Column 'Annual_Salary' -> Numeric Float")
    print("-" * 70)
    # Strip '$' and ',' using pandas string methods
    df['Annual_Salary'] = df['Annual_Salary'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
    df['Annual_Salary'] = pd.to_numeric(df['Annual_Salary'], errors='coerce')
    print("Cleaned Salary Column Data Type:", df['Annual_Salary'].dtype)
    
    # -------------------------------------------------------------------------
    # STEP 4: OUTLIER CAPPING VIA IQR (INTERQUARTILE RANGE)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[STEP 4] Outlier Detection & Capping for 'Age'")
    print("-" * 70)
    Q1 = df['Age'].quantile(0.25)
    Q3 = df['Age'].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    
    print(f"Age IQR: {IQR:.2f} | Upper Limit: {upper_limit:.2f} | Lower Limit: {lower_limit:.2f}")
    print(f"Capping values outside these limits (e.g., Age {df['Age'].max():.0f} -> {upper_limit:.2f})...")
    df['Age'] = df['Age'].clip(lower=lower_limit, upper=upper_limit)
    
    # -------------------------------------------------------------------------
    # STEP 5: MISSING VALUE IMPUTATION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[STEP 5] Imputing Missing Values (Univariate Strategy)")
    print("-" * 70)
    
    # Median imputation for continuous numerical columns
    age_median = df['Age'].median()
    salary_median = df['Annual_Salary'].median()
    df['Age'] = df['Age'].fillna(age_median)
    df['Annual_Salary'] = df['Annual_Salary'].fillna(salary_median)
    
    # Mode imputation for categorical text columns
    edu_mode = df['Education_Level'].mode()[0]
    city_mode = df['City'].mode()[0]
    df['Education_Level'] = df['Education_Level'].fillna(edu_mode)
    df['City'] = df['City'].fillna(city_mode)
    
    print(f"Imputed 'Age' missing values with Median: {age_median:.1f}")
    print(f"Imputed 'Annual_Salary' missing values with Median: ${salary_median:,.2f}")
    print(f"Imputed 'Education_Level' missing values with Mode: {edu_mode}")
    print(f"Imputed 'City' missing values with Mode: {city_mode}")
    print("\nRemaining Missing Values Check:", df.isnull().sum().sum())
    
    # -------------------------------------------------------------------------
    # STEP 6: CATEGORICAL ENCODING (ORDINAL & NOMINAL)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[STEP 6] Categorical Encoding (Ordinal Mapping & One-Hot Encoding)")
    print("-" * 70)
    
    # 6A: Ordinal Encoding for Education_Level
    education_mapping = {
        'High School': 0,
        'Bachelor': 1,
        'Master': 2,
        'PhD': 3
    }
    df['Education_Encoded'] = df['Education_Level'].map(education_mapping)
    print("Ordinal Encoding Map applied to Education_Level.")
    
    # 6B: Nominal One-Hot Encoding for City (Dropping first to avoid Dummy Variable Trap)
    df_encoded = pd.get_dummies(df, columns=['City'], drop_first=True, dtype=int)
    
    # Target Encoding (Purchased: No -> 0, Yes -> 1)
    df_encoded['Target'] = df_encoded['Purchased'].map({'No': 0, 'Yes': 1})
    
    # Drop non-numeric / ID columns
    features_df = df_encoded.drop(columns=['Customer_ID', 'Education_Level', 'Purchased', 'Target'])
    target = df_encoded['Target']
    
    print("\nProcessed Features Matrix (X) Sample:")
    print(features_df.head())
    
    # -------------------------------------------------------------------------
    # STEP 7: TRAIN-TEST SPLITTING
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[STEP 7] Train-Test Splitting (Preventing Overfitting)")
    print("-" * 70)
    X = features_df
    y = target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Full Dataset Rows: {len(X)}")
    print(f"Training Set (70%): {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"Testing Set (30%):  {X_test.shape[0]} rows, {X_test.shape[1]} features")
    
    # -------------------------------------------------------------------------
    # STEP 8: FEATURE SCALING (PREVENTING DATA LEAKAGE)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[STEP 8] Feature Scaling via StandardScaler (Zero Data Leakage)")
    print("-" * 70)
    
    scaler = StandardScaler()
    
    # Crucial Rule: fit_transform ONLY on Training Data!
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    num_cols = ['Age', 'Annual_Salary']
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    
    # Transform ONLY on Test Data (using Training Parameters mean and std)
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    print("\nLearned Means from Training Data:", scaler.mean_)
    print("Learned Variances from Training Data:", scaler.var_)
    
    print("\n--- Final Scaled X_train (First 3 Rows) ---")
    print(X_train_scaled.head(3))
    
    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETED SUCCESSFULLY! PREPROCESSED DATA READY FOR ML MODELS ")
    print("=" * 70)

if __name__ == '__main__':
    run_data_preprocessing_pipeline()
