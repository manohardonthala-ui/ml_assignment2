"""
Generates a synthetic Heart Failure Prediction dataset (918 rows, 11 features + target)
matching the statistical profile of the fedesoriano/heart-failure-prediction Kaggle dataset.
Run once to create heart_disease.csv.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 918

# --- numerical features ---
age      = rng.integers(28, 77, N)
resting_bp = np.clip(rng.normal(132, 18, N).astype(int), 80, 200)
cholesterol = np.where(rng.random(N) < 0.18, 0,
              np.clip(rng.normal(240, 55, N).astype(int), 85, 564))
fasting_bs = (rng.random(N) < 0.23).astype(int)
max_hr   = np.clip(rng.normal(136, 25, N).astype(int), 60, 202)
oldpeak  = np.round(np.clip(rng.normal(0.89, 1.07, N), -2.6, 6.2), 1)

# --- categorical features ---
sex_vals          = rng.choice(["M", "F"],       N, p=[0.79, 0.21])
chest_pain_vals   = rng.choice(["ATA", "NAP", "ASY", "TA"], N, p=[0.22, 0.22, 0.54, 0.02])
resting_ecg_vals  = rng.choice(["Normal", "ST", "LVH"], N, p=[0.60, 0.12, 0.28])
exercise_angina   = rng.choice(["N", "Y"],        N, p=[0.59, 0.41])
st_slope_vals     = rng.choice(["Up", "Flat", "Down"], N, p=[0.43, 0.46, 0.11])

# --- target: correlated with risk factors ---
logit = (
    -0.5
    + 0.04  * (age - 54)
    + 0.8   * (sex_vals == "M").astype(float)
    + 1.0   * (chest_pain_vals == "ASY").astype(float)
    - 0.5   * (chest_pain_vals == "ATA").astype(float)
    + 0.015 * np.maximum(0, resting_bp - 120)
    + 0.4   * fasting_bs
    - 0.012 * np.maximum(0, max_hr - 100)
    + 0.5   * oldpeak
    + 0.6   * (exercise_angina == "Y").astype(float)
    - 0.8   * (st_slope_vals == "Up").astype(float)
    + 0.5   * (st_slope_vals == "Flat").astype(float)
)
prob = 1 / (1 + np.exp(-logit))
# add noise
prob = np.clip(prob + rng.normal(0, 0.05, N), 0.02, 0.98)
heart_disease = (rng.random(N) < prob).astype(int)

df = pd.DataFrame({
    "Age": age,
    "Sex": sex_vals,
    "ChestPainType": chest_pain_vals,
    "RestingBP": resting_bp,
    "Cholesterol": cholesterol,
    "FastingBS": fasting_bs,
    "RestingECG": resting_ecg_vals,
    "MaxHR": max_hr,
    "ExerciseAngina": exercise_angina,
    "Oldpeak": oldpeak,
    "ST_Slope": st_slope_vals,
    "HeartDisease": heart_disease,
})

out = "heart_disease.csv"
df.to_csv(out, index=False)
print(f"Saved {out}  shape={df.shape}  target balance={df.HeartDisease.mean():.2f}")
