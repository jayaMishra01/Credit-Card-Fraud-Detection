import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "Fraud Detection Model.pkl"

print("=" * 60)
print("CREDIT CARD FRAUD DETECTION")
print("=" * 60)

print("\nLoading dataset...")

if not os.path.exists(DATA_PATH):
    print("ERROR: Dataset not found!")
    print("Make sure creditcard.csv is inside the data folder.")
    raise FileNotFoundError(DATA_PATH)

data = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

print("\nDataset columns:")
print(data.columns.tolist())

print("\nMissing values:")
print(data.isnull().sum().sum())

print("\nClass distribution:")
print(data["Class"].value_counts())

print("\nClass percentage:")
print(data["Class"].value_counts(normalize=True) * 100)

data = data.dropna()

print("\nDataset shape after cleaning:")
print(data.shape)

X = data.drop("Class", axis=1)
y = data["Class"]

print("\nNumber of features:", X.shape[1])
print("Target column: Class")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])

print("\nTraining model...")
model.fit(X_train, y_train)

print("Model training completed!")

print("\nMaking predictions...")
y_pred = model.predict(X_test)

print("Predictions completed!")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1-Score :", round(f1 * 100, 2), "%")

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0
    )
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nSaving trained model...")

joblib.dump(model, MODEL_PATH)

print("Model saved successfully!")
print("Saved as:", MODEL_PATH)

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)