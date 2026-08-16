# Heart Disease Classification — ML Assignment

## a. Problem Statement

Cardiovascular disease is one of the leading causes of death globally. Early and accurate prediction of heart disease can significantly improve patient outcomes. This project builds and compares five machine learning classification models to predict whether a patient has heart disease based on clinical features. The goal is to identify the best-performing model across six evaluation metrics: Accuracy, AUC, Precision, Recall, F1 Score, and MCC.

---

## b. Dataset Description

| Property           | Details                                                                 |
|--------------------|-------------------------------------------------------------------------|
| **Name**           | Heart Failure Prediction Dataset                                        |
| **Source**         | Based on fedesoriano/heart-failure-prediction (Kaggle / UCI)            |
| **Rows**           | 918                                                                     |
| **Features**       | 11 input features + 1 binary target                                     |
| **Target Column**  | `HeartDisease` (0 = No Disease, 1 = Heart Disease)                      |
| **Problem Type**   | Binary Classification                                                   |

### Feature Description

| Feature          | Type        | Description                                                           |
|------------------|-------------|-----------------------------------------------------------------------|
| Age              | Numerical   | Age of the patient (years)                                            |
| Sex              | Categorical | M = Male, F = Female                                                  |
| ChestPainType    | Categorical | TA (Typical Angina), ATA, NAP, ASY                                    |
| RestingBP        | Numerical   | Resting blood pressure (mm Hg)                                        |
| Cholesterol      | Numerical   | Serum cholesterol (mm/dl); 0 = not measured                           |
| FastingBS        | Binary      | 1 if fasting blood sugar > 120 mg/dl, else 0                          |
| RestingECG       | Categorical | Normal, ST, LVH                                                       |
| MaxHR            | Numerical   | Maximum heart rate achieved                                           |
| ExerciseAngina   | Categorical | Exercise-induced angina: Y = Yes, N = No                              |
| Oldpeak          | Numerical   | ST depression induced by exercise                                     |
| ST_Slope         | Categorical | Slope of peak exercise ST segment: Up, Flat, Down                     |
| **HeartDisease** | **Target**  | **0 = No Heart Disease, 1 = Heart Disease**                           |

---

## c. GitHub Repository Link

> **https://github.com/YOUR_USERNAME/ml_assignment**
>
> *(Replace with your actual GitHub URL after pushing)*

Repository contains:
- `app.py` — Streamlit web application
- `requirements.txt` — Python dependencies
- `README.md` — This file
- `test_data.csv` — 20% test split (184 rows) for Streamlit upload
- `model/train_models.py` — Training script
- `model/heart_disease.csv` — Full dataset (918 rows)
- `model/*.pkl` — Saved model files (5 models)

---

## d. Models Used

### Comparison Table — Evaluation Metrics

| ML Model Name              | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression        | 0.7554   | 0.7465 | 0.7808    | 0.8976 | 0.8352 | 0.3841 |
| Decision Tree              | 0.6957   | 0.6635 | 0.7983    | 0.7480 | 0.7724 | 0.3163 |
| KNN                        | 0.6739   | 0.6496 | 0.7310    | 0.8346 | 0.7794 | 0.1702 |
| Naive Bayes                | 0.7391   | 0.7536 | 0.7651    | 0.8976 | 0.8261 | 0.3341 |
| Random Forest (Ensemble)   | 0.7174   | 0.7330 | 0.7622    | 0.8583 | 0.8074 | 0.2909 |

---

### Observations on Model Performance

| ML Model Name            | Observation about model performance                                                                                                                                                                            |
|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression      | Achieved the **best overall accuracy (75.54%)** and highest MCC (0.3841), indicating strong linear separability in the feature space. High recall (0.8976) makes it excellent at catching true heart disease cases. Performs well with scaled features. |
| Decision Tree            | Showed moderate accuracy (69.57%) and the highest precision (0.7983) among all models, meaning it makes fewer false positive predictions. However, it is prone to overfitting without pruning, resulting in lower generalization compared to ensemble methods. |
| KNN                      | Achieved the lowest accuracy (67.39%) and MCC (0.1702), suggesting it struggles with the feature scale and dimensionality of this dataset. Sensitive to the choice of K and benefits from feature scaling. Good recall (0.8346) but poor AUC indicates limited discrimination ability. |
| Naive Bayes              | Performed well with accuracy of 73.91% and the **highest AUC (0.7536)**, showing good probabilistic calibration. Strong recall (0.8976) makes it useful for medical screening where missing a positive case is costly. The independence assumption does not hurt performance much here. |
| Random Forest (Ensemble) | Solid and balanced performance (accuracy 71.74%, AUC 0.7330). As an ensemble method, it is more robust to noise and overfitting than a single Decision Tree. Slightly lower accuracy than Logistic Regression on this dataset, but generally the most reliable choice in practice. |
| **Overall Winner**       | **Logistic Regression** — Best accuracy (75.54%) and MCC (0.3841). Naive Bayes closely follows with the best AUC, making it the preferred choice when probability estimation matters most. |

---

## Live Streamlit App Link

> **https://YOUR_APP_NAME.streamlit.app**
>
> *(Replace with your actual Streamlit Community Cloud URL after deployment)*

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ml_assignment.git
cd ml_assignment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (generates .pkl files and test_data.csv)
python model/train_models.py

# 4. Launch the Streamlit app
streamlit run app.py

# 5. Open browser at http://localhost:8501
#    Upload test_data.csv and explore results
```

---

## Project Structure

```
ml_assignment/
├── app.py                        ← Streamlit web application
├── requirements.txt              ← Python package dependencies
├── README.md                     ← This file
├── test_data.csv                 ← Test split for Streamlit upload (184 rows)
└── model/
    ├── generate_dataset.py       ← Dataset generation script
    ├── train_models.py           ← Model training and evaluation
    ├── heart_disease.csv         ← Full dataset (918 rows)
    ├── logistic_regression.pkl   ← Saved Logistic Regression model
    ├── decision_tree.pkl         ← Saved Decision Tree model
    ├── knn.pkl                   ← Saved KNN model
    ├── naive_bayes.pkl           ← Saved Naive Bayes model
    └── random_forest.pkl         ← Saved Random Forest model
```
