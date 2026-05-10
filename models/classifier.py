"""
NeuraX - ML Classifier
Trains an SVM (primary) + Random Forest (secondary) on EEG features.
Saves the best model to models/neurax_model.pkl
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

from data.signal_generator import generate_dataset, COMMANDS
from preprocessing.pipeline import preprocess_dataset

MODEL_PATH = os.path.join(os.path.dirname(__file__), "neurax_model.pkl")


def train(samples_per_class: int = 400, noise_level: float = 0.3):
    print("=" * 50)
    print("  NeuraX - Model Training")
    print("=" * 50)

    # 1. Generate dataset
    print(f"\n[1/5] Generating EEG dataset ({samples_per_class} samples/class)...")
    X_raw, y = generate_dataset(samples_per_class=samples_per_class, noise_level=noise_level)
    print(f"      Raw data shape: {X_raw.shape}")

    # 2. Preprocess
    print("[2/5] Extracting features from EEG windows...")
    X = preprocess_dataset(X_raw)
    print(f"      Feature matrix shape: {X.shape}")

    # 3. Train/test split
    print("[3/5] Splitting data (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Build pipelines
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
    ])

    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])

    # 5. Train and evaluate both
    print("[4/5] Training models...")
    results = {}

    for name, pipe in [("SVM (RBF)", svm_pipeline), ("Random Forest", rf_pipeline)]:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
        results[name] = {"pipeline": pipe, "accuracy": acc, "cv_mean": cv_scores.mean()}
        print(f"\n  [{name}]")
        print(f"    Test Accuracy : {acc * 100:.2f}%")
        print(f"    CV Mean       : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

    # Pick best model
    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    best = results[best_name]
    print(f"\n[5/5] Best model: {best_name} ({best['cv_mean'] * 100:.2f}% CV)")

    # Detailed report for best model
    y_pred_best = best["pipeline"].predict(X_test)
    label_names = [COMMANDS[i] for i in sorted(COMMANDS.keys())]
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred_best, target_names=label_names))

    # Save model
    joblib.dump(best["pipeline"], MODEL_PATH)
    print(f"\n  Model saved to: {MODEL_PATH}")
    print("=" * 50)

    return best["pipeline"]


def load_model():
    """Load the saved model. Trains a new one if not found."""
    if not os.path.exists(MODEL_PATH):
        print("No saved model found. Training now...")
        return train()
    return joblib.load(MODEL_PATH)


def predict(signal_raw: np.ndarray, model=None) -> dict:
    """
    Predict command from a raw EEG window.
    Returns dict with command name and confidence scores.
    """
    from preprocessing.pipeline import extract_features

    if model is None:
        model = load_model()

    features = extract_features(signal_raw).reshape(1, -1)
    command_id = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]

    return {
        "command_id": command_id,
        "command": COMMANDS[command_id],
        "confidence": float(proba[command_id]),
        "all_probabilities": {
            COMMANDS[i]: float(p) for i, p in enumerate(proba)
        },
    }


if __name__ == "__main__":
    train()
