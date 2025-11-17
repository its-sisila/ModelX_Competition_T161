# Dementia Prediction from Non-Medical Features (ModelX Sprint)
**Team:** skill_issues (T161)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/pandas-2.x-blue?logo=pandas)
![Google Colab](https://img.shields.io/badge/Google%20Colab-f9ab00?logo=googlecolab)

This repository contains the complete solution for the **ModelX Optimization Sprint**. The objective was to build a binary classification model to predict dementia risk using *only* non-medical data from the NACC dataset.

---

## 1. Project Goal & The Challenge

The primary goal was to explore how well dementia risk could be predicted using only demographic and non-medical information, completely excluding all cognitive tests, medical history, and biomarker data.

* **The Constraint:** All features related to medical diagnoses, cognitive scores (e.g., `MMSE`, `MOCA`), or biological markers (e.g., CSF, PET) were strictly forbidden.
* **The Data:** A large dataset (`Dementia Prediction Dataset.csv`) from the National Alzheimer's Coordinating Center (NACC).
* **The Target:** The model predicts the `NACCALZD` variable, where:
    * **Class 0 (Normal):** `NACCALZD` == 8 ("No cognitive impairment")
    * **Class 1 (Dementia):** `NACCALZD` == 1 ("Yes", dementia)

---

## 2. Final Model & Key Results

A **Tuned Random Forest Classifier** was selected as the final model, as it provided a **22.2% F1-Score improvement** over the baseline Logistic Regression.

| Model | Accuracy | F1-Score (Dementia) |
| :--- | ---: | ---: |
| Logistic Regression (Baseline) | 0.6199 | 0.4226 |
| **Tuned Random Forest (Final)** | **0.6438** | **0.5164** |

The final model's hyperparameters were: `{'model__max_depth': 20, 'model__min_samples_leaf': 2, 'model__n_estimators': 200}`.

### Key Feature Drivers
The model's predictions are dominated by two key features, which together account for **over 75%** of the predictive power:

1.  **`AGE` (42.4%):** The most significant predictor.
2.  **`EDUC` (33.2%):** Years of education was the second most significant predictor.

---

## 3. How to Run This Project

This project is designed to be run in a **Google Colab** environment.

1.  **Upload Data:**
    * Upload your `Dementia Prediction Dataset.csv` file to your Google Drive.
    * The code assumes this file is located at: `/content/drive/MyDrive/Colab Notebooks/Dementia Prediction Dataset.csv`.
    * *If your path is different, you must edit the `filepath` variable in the script.*

2.  **Open in Colab:**
    * Upload the `modelx_colab_final_no_shap.py` script to your Google Drive and open it with Colab, or create a new Colab notebook and paste the code in.

3.  **Run the Code:**
    * Execute the script cells from top to bottom.
    * The script will first ask for permission to **mount your Google Drive**. You must authenticate this to allow the script to read your CSV file.
    * The script will then run the full pipeline, from data loading to evaluation, and print the final model comparison and feature importance tables to the console.

---

## 4. Project Workflow

The script `modelx_colab_final_no_shap.py` performs the following steps:

1.  **Load Data:** Loads only the 8 essential columns (`VISITYR`, `BIRTHYR`, `EDUC`, `SEX`, `MARISTAT`, `RACE`, `HANDED`, `NACCALZD`) to save memory.
2.  **Feature Engineering:** Creates the `AGE` feature by calculating `VISITYR - BIRTHYR`.
3.  **Target Cleaning:** Filters the `NACCALZD` target column to keep only codes `1` (Dementia) and `8` (Normal), then maps them to `1` and `0`.
4.  **Preprocessing:** Uses a `scikit-learn Pipeline` to:
    * **Impute** missing values (`median` for numerical, `most_frequent` for categorical).
    * **Scale** numerical features (`StandardScaler`).
    * **Encode** categorical features (`OneHotEncoder`).
5.  **Modeling:** Trains a `LogisticRegression` (baseline) and a `GridSearchCV`-tuned `RandomForestClassifier`.
6.  **Evaluation:** Compares the models on the `F1-Score` for the dementia class.
7.  **Explainability:** Extracts and prints the `feature_importances_` from the final Random Forest model.
