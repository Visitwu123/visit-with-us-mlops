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


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

TRAIN_PATH = "tourism_project/data/train.csv"
TEST_PATH = "tourism_project/data/test.csv"
MODEL_PATH = "tourism_project/model_building/best_model.joblib"
TARGET = "ProdTaken"


# --------------------------------------------------
# CHECK FILES
# --------------------------------------------------

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(TRAIN_PATH)

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(TEST_PATH)


# --------------------------------------------------
# LOAD TRAIN AND TEST DATA
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training data shape:", train_df.shape)
print("Testing data shape :", test_df.shape)


# --------------------------------------------------
# SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


# --------------------------------------------------
# IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# --------------------------------------------------

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# --------------------------------------------------
# NUMERICAL PREPROCESSING
# --------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


# --------------------------------------------------
# CATEGORICAL PREPROCESSING
# --------------------------------------------------

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# --------------------------------------------------
# COMBINE PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# --------------------------------------------------
# RANDOM FOREST MODEL
# --------------------------------------------------

rf_model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)


# --------------------------------------------------
# COMPLETE PIPELINE
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf_model)
    ]
)


# --------------------------------------------------
# HYPERPARAMETER GRID
# --------------------------------------------------

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}


# --------------------------------------------------
# MLflow EXPERIMENT
# --------------------------------------------------

mlflow.set_experiment(
    "Visit-With-Us-Random-Forest"
)


# --------------------------------------------------
# MODEL TRAINING AND EXPERIMENT TRACKING
# --------------------------------------------------

with mlflow.start_run(
    run_name="Random-Forest-Tuning"
):

    # Log basic parameters

    mlflow.log_param(
        "algorithm",
        "Random Forest"
    )

    mlflow.log_param(
        "cv_folds",
        3
    )


    # --------------------------------------------------
    # GRID SEARCH
    # --------------------------------------------------

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )


    # Train the model

    grid_search.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------
    # GET BEST MODEL AND PARAMETERS
    # --------------------------------------------------

    best_model = grid_search.best_estimator_

    best_params = grid_search.best_params_

    best_cv_f1 = grid_search.best_score_


    # --------------------------------------------------
    # DISPLAY BEST PARAMETERS
    # --------------------------------------------------

    print("\nBest Parameters:")
    print(best_params)

    print("\nBest Cross-Validation F1 Score:")
    print(best_cv_f1)


    # --------------------------------------------------
    # LOG BEST PARAMETERS
    # --------------------------------------------------

    mlflow.log_params(
        best_params
    )

    mlflow.log_metric(
        "best_cv_f1",
        best_cv_f1
    )


    # --------------------------------------------------
    # TEST SET PREDICTION
    # --------------------------------------------------

    y_pred = best_model.predict(
        X_test
    )


    # --------------------------------------------------
    # EVALUATION METRICS
    # --------------------------------------------------

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


    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print("\nTest Set Performance")
    print("-" * 40)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


    # --------------------------------------------------
    # LOG TEST METRICS TO MLFLOW
    # --------------------------------------------------

    mlflow.log_metric(
        "test_accuracy",
        accuracy
    )

    mlflow.log_metric(
        "test_precision",
        precision
    )

    mlflow.log_metric(
        "test_recall",
        recall
    )

    mlflow.log_metric(
        "test_f1",
        f1
    )


    # --------------------------------------------------
    # LOG MODEL TO MLFLOW
    # --------------------------------------------------

    mlflow.sklearn.log_model(
        best_model,
        "random_forest_model"
    )


    # --------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    print("\nBest model saved successfully:")

    print(
        MODEL_PATH
    )
