"""
train_models.py
Train 5 classification models on the Heart Failure Prediction dataset,
evaluate with 6 metrics, save models as .pkl, and export test_data.csv.
"""
import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "heart_disease.csv")
OUT_DIR    = SCRIPT_DIR  # save .pkl files here
TEST_CSV   = os.path.join(os.path.dirname(SCRIPT_DIR), "test_data.csv")


# ── load & preprocess ────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    return X, y, encoders


# ── metrics helper ───────────────────────────────────────────────────────────
def compute_metrics(model, X_test, y_test, needs_proba=True):
    y_pred = model.predict(X_test)
    if needs_proba and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred),         4),
        "AUC":       round(roc_auc_score(y_test, y_prob),          4),
        "Precision": round(precision_score(y_test, y_pred),        4),
        "Recall":    round(recall_score(y_test, y_pred),           4),
        "F1":        round(f1_score(y_test, y_pred),               4),
        "MCC":       round(matthews_corrcoef(y_test, y_pred),      4),
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    X, y, encoders = load_data()

    # scale for LR and KNN
    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # save test data (unscaled features + true label) for Streamlit upload
    test_df = X_test.copy()
    test_df["HeartDisease"] = y_test.values
    test_df.to_csv(TEST_CSV, index=False)
    print(f"Saved test_data.csv  ({len(test_df)} rows)")

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            X_train_s, X_test_s,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            X_train, X_test,
        ),
        "KNN": (
            KNeighborsClassifier(n_neighbors=5),
            X_train_s, X_test_s,
        ),
        "Naive Bayes": (
            GaussianNB(),
            X_train, X_test,
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, random_state=42),
            X_train, X_test,
        ),
    }

    results = {}
    pkl_map = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree":       "decision_tree.pkl",
        "KNN":                 "knn.pkl",
        "Naive Bayes":         "naive_bayes.pkl",
        "Random Forest":       "random_forest.pkl",
    }

    for name, (clf, Xtr, Xte) in models.items():
        clf.fit(Xtr, y_train)
        metrics = compute_metrics(clf, Xte, y_test)
        results[name] = metrics

        # save model (wrap scaled models with scaler inside a dict)
        payload = {"model": clf}
        if name in ("Logistic Regression", "KNN"):
            payload["scaler"] = scaler
        payload["encoders"] = encoders
        pkl_path = os.path.join(OUT_DIR, pkl_map[name])
        joblib.dump(payload, pkl_path)
        print(f"  Saved {pkl_path}")

    # ── print comparison table ──────────────────────────────────────────────
    metrics_order = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    col_w = 20
    header = f"{'Model':<25}" + "".join(f"{m:>{col_w}}" for m in metrics_order)
    print("\n" + "=" * (25 + col_w * len(metrics_order)))
    print("CLASSIFICATION MODEL COMPARISON")
    print("=" * (25 + col_w * len(metrics_order)))
    print(header)
    print("-" * (25 + col_w * len(metrics_order)))
    for model_name, mvals in results.items():
        row = f"{model_name:<25}" + "".join(f"{mvals[m]:>{col_w}}" for m in metrics_order)
        print(row)
    print("=" * (25 + col_w * len(metrics_order)))


if __name__ == "__main__":
    main()
