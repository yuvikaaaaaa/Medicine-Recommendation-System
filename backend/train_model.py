#!/usr/bin/env python3
"""
Train the disease prediction model using Training.csv
Run: python train_model.py
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "ml")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_training_data():
    path = os.path.join(DATA_DIR, "Training.csv")
    print("PATH =", path)
    df = pd.read_csv(path)
    # Drop unnamed columns if any
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


def train():
    print("Loading training data...")
    df = load_training_data()

    print(f"Dataset shape: {df.shape}")
    print(f"Diseases: {df['prognosis'].nunique()}")

    # Features and target
    X = df.drop("prognosis", axis=1)
    y = df["prognosis"]

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    feature_columns = list(X.columns)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    cv_scores = cross_val_score(model, X, y_encoded, cv=5, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Save model and encoders
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    encoder_path = os.path.join(MODEL_DIR, "encoders.pkl")
    features_path = os.path.join(MODEL_DIR, "features.pkl")

    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    joblib.dump(feature_columns, features_path)

    print(f"Model saved: {model_path}")
    print(f"Encoders saved: {encoder_path}")
    print(f"Features saved: {features_path}")
    print("Training complete!")

    return model, le, feature_columns


if __name__ == "__main__":
    train()
