# classification.py
# Tourism Experience Analytics
# Objective: Predict VisitMode

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import joblib


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "processed",
    "master_tourism_data.csv"
)

DATA_PATH = os.path.abspath(DATA_PATH)

print("=" * 60)
print("TOURISM EXPERIENCE ANALYTICS - CLASSIFICATION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

features = [
    "VisitYear",
    "VisitMonth",
    "AttractionId",
    "AttractionTypeId",
    "AttractionCityId",
    "ContinentId",
    "RegionId",
    "CountryId",
    "CityId"
]

target = "VisitMode"


# ============================================================
# 3. PREPARE DATA
# ============================================================

clf_df = df[features + [target]].copy()

# CityId contains a few missing values
clf_df["CityId"] = clf_df["CityId"].fillna(-1)

# Remove any remaining missing values
clf_df = clf_df.dropna()

print("\nClassification dataset shape:", clf_df.shape)

print("\nVisitMode distribution:")
print(clf_df[target].value_counts())


# ============================================================
# 4. SPLIT FEATURES AND TARGET
# ============================================================

X = clf_df[features]
y = clf_df[target]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. PREPROCESSING
# ============================================================

categorical_features = [
    "AttractionId",
    "AttractionTypeId",
    "AttractionCityId",
    "ContinentId",
    "RegionId",
    "CountryId",
    "CityId"
]

numerical_features = [
    "VisitYear",
    "VisitMonth"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================================
# 7. DEFINE CLASSIFICATION MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
}


# ============================================================
# 8. TRAIN AND EVALUATE MODELS
# ============================================================

results = {}
trained_models = {}

for model_name, model in models.items():

    print("\n" + "-" * 60)
    print("Training:", model_name)
    print("-" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    trained_models[model_name] = pipeline

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results).T

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)


# ============================================================
# 10. SELECT BEST MODEL
# ============================================================

best_model_name = results_df["F1 Score"].idxmax()

best_model = trained_models[best_model_name]

print("\n" + "=" * 60)
print("BEST CLASSIFICATION MODEL")
print("=" * 60)

print("Best model:", best_model_name)
print("Best F1 Score:", results_df.loc[best_model_name, "F1 Score"])


# ============================================================
# 11. DETAILED CLASSIFICATION REPORT
# ============================================================

y_pred_best = best_model.predict(X_test)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred_best,
        zero_division=0
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred_best)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# 13. SAVE BEST MODEL
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "visit_mode_model.pkl"
)

joblib.dump(best_model, MODEL_PATH)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print("Saved to:", MODEL_PATH)

print("\nClassification completed successfully!")