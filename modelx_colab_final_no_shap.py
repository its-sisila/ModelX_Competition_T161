# --- 1. SETUP AND IMPORT LIBRARIES ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import shap # SHAP IS REMOVED
import warnings

# Sklearn imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
print("--- 1. Libraries Imported ---")

# --- 2. MOUNT GOOGLE DRIVE ---
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    print("--- Google Drive Mounted Successfully ---")
except ImportError:
    print("--- Not running in Colab, assuming file is local ---")

# --- 3. LOAD DATA ---
filepath = '/content/drive/MyDrive/Colab Notebooks/Dementia Prediction Dataset.csv'

columns_to_load = [
    'VISITYR',  # Needed to calculate AGE
    'BIRTHYR',  # Needed to calculate AGE
    'EDUC',     # Education
    'SEX',      # Sex
    'MARISTAT', # Marital Status (was 'MARRIED')
    'RACE',     # Race
    'HANDED',   # Handedness (was 'HAND')
    'NACCALZD'  # THE REAL TARGET (was 'NACCADC' or 'DX')
]

try:
    df = pd.read_csv(filepath, low_memory=False, usecols=columns_to_load)
    print(f"--- 3. Successfully loaded required columns from: {filepath} ---")
except FileNotFoundError:
    print(f"ERROR: Could not find file at '{filepath}'.")
    exit()
except ValueError as e:
    print(f"ERROR: A column might be misspelled. Check list: {columns_to_load}")
    print(f"Full error: {e}")
    exit()
except Exception as e:
    print(f"An error occurred during file loading: {e}")
    exit()


# --- 4. FEATURE ENGINEERING & SELECTION ---

print("--- 4. Engineering 'AGE' feature (VISITYR - BIRTHYR) ---")
df['AGE'] = df['VISITYR'] - df['BIRTHYR']

allowed_features = [
    'AGE',      # Our new engineered feature
    'EDUC',
    'SEX',
    'MARISTAT',
    'RACE',
    'HANDED'
]
target_variable = 'NACCALZD'
print("--- 4. Feature Engineering Complete. ---")

# --- 5. DATA CLEANING & PREPROCESSING ---
print("--- 5. Cleaning Data... ---")

codes_normal = [8]      # 8 = No cognitive impairment
codes_dementia = [1]    # 1 = Yes (Dementia)

df_filtered = df[df[target_variable].isin(codes_normal + codes_dementia)].copy()
df_filtered['target'] = df_filtered[target_variable].map(lambda x: 0 if x in codes_normal else 1)

print("\nFiltered Target value counts (binary):")
print(df_filtered['target'].value_counts())

df_filtered[allowed_features] = df_filtered[allowed_features].replace(-4, np.nan)

X = df_filtered[allowed_features]
y = df_filtered['target']

numeric_features = ['AGE', 'EDUC']
categorical_features = ['SEX', 'MARISTAT', 'RACE', 'HANDED']

print("--- Data Cleaned and Processed. ---")

# --- 6. BUILD PREPROCESSING PIPELINES ---
print("--- 6. Building Preprocessing Pipelines... ---")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# --- 7. TRAIN-TEST SPLIT ---
print("--- 7. Splitting Data into Train/Test... ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# --- 8. MODEL BUILDING ---

print("\nTraining Logistic Regression...")
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(random_state=42, solver='liblinear'))
])
lr_pipeline.fit(X_train, y_train)

print("\nTraining and Tuning Random Forest (GridSearchCV)...")
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20],
    'model__min_samples_leaf': [2, 4]
}

grid_search = GridSearchCV(
    estimator=rf_pipeline, 
    param_grid=param_grid, 
    cv=5, 
    n_jobs=-1, 
    scoring='f1',
    verbose=1
)
grid_search.fit(X_train, y_train)

final_model = grid_search.best_estimator_
print(f"\nBest Random Forest Params: {grid_search.best_params_}")
print("--- 8. Model training complete. ---")

# --- 9. MODEL EVALUATION ---

print("\n\n" + "="*30)
print("   MODEL EVALUATION RESULTS   ")
print("="*30 + "\n")

print("--- Logistic Regression (Baseline) ---")
y_pred_lr = lr_pipeline.predict(X_test)
print(classification_report(y_test, y_pred_lr, target_names=['Normal (0)', 'Dementia (1)']))
f1_lr = f1_score(y_test, y_pred_lr, pos_label=1)

print("--- Tuned Random Forest (Final Model) ---")
y_pred_final = final_model.predict(X_test)
print(classification_report(y_test, y_pred_final, target_names=['Normal (0)', 'Dementia (1)']))
f1_rf = f1_score(y_test, y_pred_final, pos_label=1)

print("\n--- Model Comparison ---")
evaluation_data = {
    "Metric": ["Accuracy", "F1-Score (Dementia)"],
    "Logistic Regression": [
        accuracy_score(y_test, y_pred_lr),
        f1_lr
    ],
    "Tuned Random Forest": [
        accuracy_score(y_test, y_pred_final),
        f1_rf
    ]
}
eval_df = pd.DataFrame(evaluation_data).set_index("Metric")
print(eval_df.to_markdown(floatfmt=".4f"))

# --- 10. NEW: VERBAL EXPLAINABILITY ---
print("\n--- 10. Generating Feature Importances ---")

try:
    # Get the pipeline steps
    preprocessor_step = final_model.named_steps['preprocessor']
    model_step = final_model.named_steps['model']
    
    # Get the feature names from the OneHotEncoder
    ohe_feature_names = preprocessor_step.named_transformers_['cat'] \
        .named_steps['onehot'] \
        .get_feature_names_out(categorical_features)
    
    # Combine all feature names in the correct order
    all_feature_names = numeric_features + list(ohe_feature_names)
    
    # Get the importances from the trained model
    importances = model_step.feature_importances_
    
    # Create a simple DataFrame to show the results
    feature_importance_df = pd.DataFrame(
        list(zip(all_feature_names, importances)),
        columns=['Feature', 'Importance']
    ).sort_values(by='Importance', ascending=False)

    print("Feature Importances (from final Random Forest model):")
    print(feature_importance_df.to_markdown(index=False, floatfmt=".5f"))

except Exception as e:
    print(f"Could not calculate feature importances: {e}")

print("\n--- PROJECT COMPLETE ---")
