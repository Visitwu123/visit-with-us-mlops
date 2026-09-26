import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


TRAIN_PATH = "tourism_project/data/train.csv"
TEST_PATH = "tourism_project/data/test.csv"

MODEL_PATH = "tourism_project/model_building/best_model.joblib"

TARGET = "ProdTaken"


# Load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training data:", train_df.shape)
print("Testing data :", test_df.shape)


# Separate features and target
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


# Identify columns
numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()


# Preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# Define Random Forest
rf_model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)


# Complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf_model)
    ]
)


# Hyperparameters
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

print("Random Forest model and parameters defined.")
print(param_grid)
import sys
import os

# Ensure the current working directory is in sys.path so Python can find 'tourism_project' as a package.
# The current working directory is expected to be '/content/visit-with-us-mlops'.
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# Import the necessary variables from the model_train module.
# This will execute the top-level code in model_train.py, including data loading and preprocessing.
from tourism_project.model_building.model_train import pipeline, param_grid, X_train, y_train

print("Variables (pipeline, param_grid, X_train, y_train) imported successfully.")

# MLflow experiment tracking

mlflow.set_experiment(
    "Visit-With-Us-Random-Forest"
)

with mlflow.start_run(
    run_name="Random-Forest-Tuning"
):

    mlflow.log_param(
        "algorithm",
        "Random Forest"
    )

    mlflow.log_param(
        "cv_folds",
        3
    )

    mlflow.log_params(best_params)

    mlflow.log_metric(
        "best_cv_f1",
        best_cv_f1
    )

    print("\nParameters logged to MLflow.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Test set evaluation

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nTest Set Performance")
print("-" * 40)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)
from tourism_project.model_building.model_train import MODEL_PATH

# Log the trained model to MLflow

mlflow.sklearn.log_model(
    best_model,
    "random_forest_model"
)


# Save best model

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nBest model saved successfully:")
print(MODEL_PATH)
