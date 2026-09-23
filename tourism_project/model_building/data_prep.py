import pandas as pd
import os
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

DATA_PATH = "tourism_project/data/tourism.csv"

TRAIN_PATH = "tourism_project/data/train.csv"

TEST_PATH = "tourism_project/data/test.csv"


# ---------------------------------------------------------
# Check whether dataset exists
# ---------------------------------------------------------

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}"
    )


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Original dataset shape:", df.shape)


# ---------------------------------------------------------
# Remove unnecessary columns
# ---------------------------------------------------------

columns_to_remove = [
    "Unnamed: 0",
    "CustomerID"
]

existing_columns = [
    col for col in columns_to_remove
    if col in df.columns
]

df = df.drop(columns=existing_columns)

print("\nColumns removed:")
print(existing_columns)

print("\nDataset shape after removing unnecessary columns:")
print(df.shape)


# ---------------------------------------------------------
# Remove duplicate records
# ---------------------------------------------------------

duplicate_count = df.duplicated().sum()

print("\nNumber of duplicate records:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()

    print(
        "Dataset shape after removing duplicates:",
        df.shape
    )
else:
    print("No duplicate records found.")


# ---------------------------------------------------------
# Check missing values
# ---------------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# ---------------------------------------------------------
# Separate features and target
# ---------------------------------------------------------

X = df.drop("ProdTaken", axis=1)

y = df["ProdTaken"]


# ---------------------------------------------------------
# Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# Re-create train and test DataFrames
# ---------------------------------------------------------

train_df = X_train.copy()

train_df["ProdTaken"] = y_train


test_df = X_test.copy()

test_df["ProdTaken"] = y_test


# ---------------------------------------------------------
# Save train and test datasets
# ---------------------------------------------------------

train_df.to_csv(
    TRAIN_PATH,
    index=False
)

test_df.to_csv(
    TEST_PATH,
    index=False
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\nData preparation completed successfully.")

print("\nTraining dataset shape:")
print(train_df.shape)

print("\nTesting dataset shape:")
print(test_df.shape)

print("\nTraining target distribution:")
print(train_df["ProdTaken"].value_counts())

print("\nTesting target distribution:")
print(test_df["ProdTaken"].value_counts())

print("\nFiles created successfully:")

print(TRAIN_PATH)

print(TEST_PATH)
