"""
app.py  –  Heart Disease Classification  |  BITS Pilani ML Assignment
Streamlit interactive app: models trained at startup (cached), upload test CSV,
select model, view metrics + confusion matrix.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
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
    confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")

CAT_COLS   = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
TARGET_COL = "HeartDisease"
DATA_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "heart_disease.csv")

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "KNN (K-Nearest Neighbor)",
    "Naive Bayes (Gaussian)",
    "Random Forest (Ensemble)",
]

# ── train all models once and cache ─────────────────────────────────────────
@st.cache_resource
def train_all_models():
    df = pd.read_csv(DATA_PATH)

    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    feature_cols = list(X_train.columns)

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols
    )

    clfs = {
        "Logistic Regression":      LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":            DecisionTreeClassifier(random_state=42),
        "KNN (K-Nearest Neighbor)": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes (Gaussian)":   GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    needs_scale = {"Logistic Regression", "KNN (K-Nearest Neighbor)"}

    trained = {}
    for name, clf in clfs.items():
        Xtr = X_train_s if name in needs_scale else X_train
        clf.fit(Xtr, y_train)
        trained[name] = clf

    return trained, scaler, encoders, needs_scale, feature_cols


# ── encode uploaded test data ────────────────────────────────────────────────
def encode_features(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col in CAT_COLS:
        if col in df.columns and col in encoders:
            le = encoders[col]
            df[col] = df[col].map(
                lambda v, le=le: le.transform([v])[0] if v in le.classes_ else 0
            )
    return df


# ── metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred),                    4),
        "AUC":       round(roc_auc_score(y_true, y_prob),                     4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0),  4),
        "Recall":    round(recall_score(y_true, y_pred),                      4),
        "F1 Score":  round(f1_score(y_true, y_pred),                          4),
        "MCC":       round(matthews_corrcoef(y_true, y_pred),                 4),
    }


def predict(clf, X_enc, scaler, name, needs_scale, feature_cols):
    if name in needs_scale:
        Xi = pd.DataFrame(scaler.transform(X_enc), columns=feature_cols)
    else:
        Xi = X_enc[feature_cols]
    y_pred = clf.predict(Xi)
    y_prob = (
        clf.predict_proba(Xi)[:, 1]
        if hasattr(clf, "predict_proba")
        else y_pred.astype(float)
    )
    return y_pred, y_prob


# ── confusion matrix plot ────────────────────────────────────────────────────
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Disease (0)", "Heart Disease (1)"],
        yticklabels=["No Disease (0)", "Heart Disease (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# PAGE
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Heart Disease Classifier", page_icon="❤️", layout="wide")

st.title("❤️  Heart Disease Classification")
st.markdown(
    """
**BITS Pilani ML Assignment** — Binary classification on the Heart Failure Prediction dataset
*918 rows · 11 features · target: HeartDisease (0 = No Disease, 1 = Heart Disease)*

Upload `test_data.csv` (from the GitHub repo), pick a model, and explore metrics.
"""
)
st.divider()

# ── load models ──────────────────────────────────────────────────────────────
with st.spinner("Training models (first load only — cached after that)…"):
    trained_models, scaler, encoders, needs_scale, feature_cols = train_all_models()

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️  Controls")
    selected_model = st.selectbox("Select Classification Model", MODEL_NAMES)
    st.markdown("---")
    st.markdown("**All 5 models available:**")
    for m in MODEL_NAMES:
        st.markdown(f"- {m}")
    st.markdown("---")
    st.info("Upload `test_data.csv` from the repo to evaluate.")

# ── a. CSV upload ─────────────────────────────────────────────────────────────
st.subheader("📂  a. Upload Test Dataset (CSV)")
uploaded = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"],
    help="Upload the test split CSV that contains feature columns + HeartDisease target.",
)

if uploaded is None:
    st.info("👆 Please upload **test_data.csv** to get started.")
    st.stop()

try:
    df_test = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if TARGET_COL not in df_test.columns:
    st.error(f"CSV must contain a `{TARGET_COL}` column.")
    st.stop()

y_true = df_test[TARGET_COL]
X_raw  = df_test.drop(columns=[TARGET_COL])
X_enc  = encode_features(X_raw, encoders)

st.success(f"Loaded {len(df_test)} rows · {X_raw.shape[1]} features")
with st.expander("Preview uploaded data (first 5 rows)"):
    st.dataframe(df_test.head(), use_container_width=True)

st.divider()

# ── b. model selection already in sidebar — show selected result ─────────────
st.subheader(f"📊  b–d.  Results for: **{selected_model}**")

clf = trained_models[selected_model]
y_pred, y_prob = predict(clf, X_enc, scaler, selected_model, needs_scale, feature_cols)
metrics = compute_metrics(y_true, y_pred, y_prob)

# ── c. evaluation metrics ─────────────────────────────────────────────────────
st.markdown("#### c. Evaluation Metrics")
c1, c2, c3, c4, c5, c6 = st.columns(6)
for col, (k, v) in zip([c1, c2, c3, c4, c5, c6], metrics.items()):
    col.metric(k, f"{v:.4f}")

st.divider()

# ── d. confusion matrix + classification report ───────────────────────────────
st.markdown("#### d. Confusion Matrix & Classification Report")
col_l, col_r = st.columns([1, 1])

with col_l:
    fig = plot_cm(y_true, y_pred, f"Confusion Matrix — {selected_model}")
    st.pyplot(fig)

with col_r:
    report = classification_report(
        y_true, y_pred,
        target_names=["No Disease (0)", "Heart Disease (1)"],
        output_dict=True,
    )
    st.dataframe(
        pd.DataFrame(report).transpose().round(4),
        use_container_width=True,
    )

st.divider()

# ── all models comparison ─────────────────────────────────────────────────────
st.subheader("📈  All Models Comparison")

rows = []
for name in MODEL_NAMES:
    c = trained_models[name]
    yp, ypr = predict(c, X_enc, scaler, name, needs_scale, feature_cols)
    m = compute_metrics(y_true, yp, ypr)
    m["Model"] = name
    rows.append(m)

metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
compare_df  = pd.DataFrame(rows)[["Model"] + metric_cols]

def highlight_max(s):
    return [
        "background-color: #d4f5d4; font-weight: bold" if v == s.max() else ""
        for v in s
    ]

st.dataframe(
    compare_df.style
    .apply(highlight_max, subset=metric_cols)
    .format({c: "{:.4f}" for c in metric_cols}),
    use_container_width=True,
    height=250,
)

# bar charts
for metric, title_suffix in [("Accuracy", "Accuracy"), ("AUC", "AUC Score")]:
    st.markdown(f"#### {title_suffix} Comparison")
    fig_b, ax_b = plt.subplots(figsize=(9, 4))
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]
    bars = ax_b.bar(compare_df["Model"], compare_df[metric],
                    color=colors, edgecolor="white", linewidth=0.8)
    ax_b.set_ylim(0, 1.1)
    ax_b.set_ylabel(metric, fontsize=11)
    ax_b.set_title(f"Model {title_suffix} Comparison", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, compare_df[metric]):
        ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                  f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_b.tick_params(axis="x", labelsize=8)
    fig_b.tight_layout()
    st.pyplot(fig_b)

st.divider()
st.caption("Heart Disease Classification · BITS Pilani ML Assignment · Dataset: Heart Failure Prediction (918 rows, 11 features)")
