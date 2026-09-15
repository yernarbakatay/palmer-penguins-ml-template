"""
Palmer Penguins Species Classifier & Data Lifecycle Training Pipeline
Template Repository for MSU AI Club Workshop 01

DATA DOWNLOAD INSTRUCTIONS:
This script automatically downloads the Palmer Penguins dataset from the public URL below.
Never commit raw dataset files directly to git repositories!

Public Data URL:
    https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Ensure cross-platform UTF-8 terminal encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Public Dataset URL
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"

def load_data(url=DATA_URL):
    """
    STAGE 1: DATA INGESTION VIA PUBLIC URL
    --------------------------------------
    Fetch dataset dynamically from remote URL.
    Never commit raw data CSV files directly to git repositories!
    """
    print(f"[Stage 1: Ingestion] Downloading dataset from public URL: {url}")
    
    # TODO: Load dataset from url into pandas DataFrame variable 'df'
    df = pd.read_csv(url) # <--- YOUR CODE HERE (e.g. pd.read_csv(url))
    
    if df is not None:
        print(f"                     Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"                     Missing values count by column:")
        
        # TODO: Calculate missing values for each column using df.isnull().sum()
        missing = df.isnull().sum()  # <--- YOUR CODE HERE
        
        if missing is not None:
            for col, null_count in missing.items():
                print(f"                       * {col:20s}: {null_count} missing values")
    else:
        print("                     ⚠️ Complete 'df = pd.read_csv(url)' above!")
        
    return df

def clean_data(df):
    """
    STAGE 2: DATA PROCESSING & IMPUTATION
    -------------------------------------
    Perform median imputation for continuous features and mode imputation for categorical attributes.
    """
    if df is None:
        print("⚠️ Data is None. Complete Stage 1 first!")
        return None

    df_clean = df.copy()
    numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    
    # TODO: Loop through numeric_cols and fill missing NaN values with median value
    for col in numeric_cols:
        # YOUR CODE HERE: Compute median and call fillna()
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
        # pass
        
    # TODO: Fill missing 'sex' column with the mode (most frequent value)
    # YOUR CODE HERE: fillna with df_clean['sex'].mode()[0]
    df_clean['sex'] = df_clean['sex'].fillna(df_clean['sex'].mode()[0])

    return df_clean

def train_model(df):
    """
    STAGE 3 & 4: FEATURE ENGINEERING, TRAINING & COHORT EVALUATION
    --------------------------------------------------------------
    Evaluate per-class precision and recall using stratified train/test split.
    """
    if df is None:
        print("⚠️ Data is None. Complete Stage 2 first!")
        return None, 0.0

    feature_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    
    # TODO: Assign feature matrix X and target y
    X = df[feature_cols]  # <--- YOUR CODE HERE (e.g. df[feature_cols])
    y = df['species']  # <--- YOUR CODE HERE (e.g. df["species"])
    
    # TODO: Split into train/test subsets using train_test_split (test_size=0.2, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # TODO: Instantiate and train RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    accuracy = 0.0
    if clf is not None and hasattr(clf, "classes_") and X_test is not None and y_test is not None:
        # TODO: Generate predictions y_pred and compute accuracy_score
        y_pred = clf.predict(X_test)

        if y_pred is not None:
            accuracy = accuracy_score(y_test, y_pred)
            print("\n" + "=" * 55)
            print(f"[Stage 4: Evaluation] Overall Model Accuracy: {accuracy:.1%}")
            print("=" * 55)
            print("\n--- Per-Species Classification Report ---")
            print(classification_report(y_test, y_pred))
            
            print("--- Feature Importance Ranks ---")
            for feat, imp in zip(feature_cols, clf.feature_importances_):
                print(f"  * {feat:20s}: {imp:.4f}")
    else:
        print("\n💡 Complete the train_test_split and RandomForestClassifier TODOs above to train your model!")
        
    return clf, accuracy

def export_model(model, output_path="penguin_model.pkl"):
    """
    STAGE 5: ARTIFACT DEPLOYMENT & EXPORT
    -------------------------------------
    Serialize trained model to file for real-time CLI and web app predictions.
    """
    if model is not None and hasattr(model, "classes_"):
        # TODO: Serialize trained model using pickle.dump()
        # YOUR CODE HERE:
        with open(output_path, "wb") as f:
            pickle.dump(model, f)
        print(f"\n[Stage 5: Export] Serialized model saved to: {output_path}")
    else:
        print("\n[Stage 5: Export] 💡 Train your model in Stage 4 before serializing!")

def main():
    print("Starting Palmer Penguins ML Classifier & Data Lifecycle Training...\n")
    df_raw = load_data()
    df_clean = clean_data(df_raw)
    model, accuracy = train_model(df_clean)
    export_model(model)
    print("\n[Success] Training Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
