import os
import joblib
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# 1. Load Dataset (UCI Breast Cancer Diagnostic: 569 samples, 30 features)
data = load_breast_cancer(as_frame=True)
df = data.frame

# Rename target for clarity
df['target'] = data.target

# 2. Train-Test Split (80/20 ratio)
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save test dataset (features + target) as test_data.csv
test_df = pd.concat([X_test, y_test], axis=1)
test_df.to_csv("../test_data.csv", index=False)

# 3. Scale Features (Fit scaler on train set)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Define Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'kNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest (Ensemble)': RandomForestClassifier(n_estimators=100, random_state=42)
}

# 5. Train, Evaluate, and Save Models
results = []
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

for name, model in models.items():
    # Fit model (scale-sensitive vs scale-insensitive)
    if name in ['Logistic Regression', 'kNN', 'Naive Bayes']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({
        'ML Model Name': name,
        'Accuracy': round(acc, 4),
        'AUC': round(auc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4)
    })

    # Save trained model artifact
    filename = f"model/model_{name.lower().replace(' ', '_').replace('_(ensemble)', '')}.pkl"
    joblib.dump(model, filename)

# Display Summary Table
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))