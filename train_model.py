
# ============================================
# CREDIT CARD FRAUD DETECTION
# Model Training Script
# ============================================

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# --------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------

DATA_PATH = "data/creditcard.csv"

print("=" * 60)
print("CREDIT CARD FRAUD DETECTION")
print("=" * 60)

print("\nLoading dataset...")

if not os.path.exists(DATA_PATH):
    print(f"ERROR: Dataset not found at {DATA_PATH}")
    print("Please make sure creditcard.csv is inside the data folder.")
    exit()

data = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# --------------------------------------------
# 2. BASIC DATA INFORMATION
# --------------------------------------------

print("\nDataset columns:")
print(data.columns.tolist())

print("\nMissing values:")
print(data.isnull().sum().sum())

print("\nClass distribution:")
print(data["Class"].value_counts())

print("\nClass percentage:")
print(data["Class"].value_counts(normalize=True) * 100)


# --------------------------------------------
# 3. REMOVE MISSING VALUES
# --------------------------------------------

data = data.dropna()

print("\nDataset shape after cleaning:", data.shape)


# --------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# --------------------------------------------

X = data.drop("Class", axis=1)
y = data["Class"]

print("\nFeatures:", X.shape[1])
print("Target column: Class")


# --------------------------------------------
# 5. TRAIN-TEST SPLIT
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------
# 6. CREATE MACHINE LEARNING PIPELINE
# --------------------------------------------

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])


# --------------------------------------------
# 7. TRAIN MODEL
# --------------------------------------------

print("\nTraining model...")
print("Please wait...")

model.fit(X_train, y_train)

print("Model training completed!")


# --------------------------------------------
# 8. MAKE PREDICTIONS
# --------------------------------------------

print("\nMaking predictions...")

y_pred = model.predict(X_test)


# --------------------------------------------
# 9. MODEL EVALUATION
# --------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    digits=4
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred
))


# --------------------------------------------
# 10. SAVE TRAINED MODEL
# --------------------------------------------

MODEL_PATH = "fraud_detection_model.pkl"

joblib.dump(model, MODEL_PATH)

print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY!")
print("=" * 60)

print(f"\nSaved model: {MODEL_PATH}")

print("\nProject structure should now be:")
print("""
Credit-Card-Fraud-Detection/
│
├── data/
│   └── creditcard.csv
│
├── train_model.py
│
└── fraud_detection_model.pkl
""")

print("\nTraining completed successfully!")
