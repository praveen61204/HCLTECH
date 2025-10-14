# Requirements: pandas, numpy, seaborn, scikit-learn, joblib
# pip install pandas numpy seaborn scikit-learn joblib

import numpy as np
import pandas as pd
import seaborn as sns            # for Titanic dataset
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from joblib import dump

# 1) Load dataset
df = sns.load_dataset("titanic")   # seaborn ships with a cleaned Titanic dataset
df = df.copy()                     # avoid modifying seaborn's original

# Keep a subset of useful columns (common example)
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
target = "survived"
df = df[features + [target]]

# Quick EDA
print("Shape:", df.shape)
print(df.head())
print("\nMissing per column:\n", df.isnull().mean())

# 2) Define X, y
X = df[features]
y = df[target]

# 3) Train-test split (important: stratify on target when classification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Build preprocessing pipelines
numeric_features = ["age", "sibsp", "parch", "fare"]
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),   # fill missing with median
    ("scaler", StandardScaler())                     # standardize numeric features
])

categorical_features = ["sex", "embarked", "pclass"]  # treat pclass as categorical
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # fill missing with mode
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 5) Full pipeline with a model (Logistic Regression here)
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(max_iter=1000))
])

# 6) Fit
pipeline.fit(X_train, y_train)

# 7) Evaluate
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]  # probability for ROC AUC

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# 8) Inspect transformed feature names (optional, useful for feature importance)
try:
    # scikit-learn >= 1.0: get_feature_names_out
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
except Exception:
    # fallback - manual composition (works in many versions)
    cat_ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"] \
              .named_steps["onehot"]
    cat_names = list(cat_ohe.get_feature_names_out(categorical_features))
    feature_names = numeric_features + cat_names

print("\nTransformed feature names:\n", feature_names)

# 9) Save pipeline for later reuse
dump(pipeline, "titanic_preproc_pipeline.joblib")
print("\nSaved pipeline to titanic_preproc_pipeline.joblib")
