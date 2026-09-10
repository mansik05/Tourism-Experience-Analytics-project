# regression.py
# Tourism Experience Analytics
# Objective: Predict Attraction Rating

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error
)


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
print("TOURISM EXPERIENCE ANALYTICS - REGRESSION")
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
    "VisitMode",
    "AttractionId",
    "AttractionTypeId",
    "AttractionCityId",
    "ContinentId",
    "RegionId",
    "CountryId",
    "CityId"
]

target = "Rating"


# ============================================================
# 3. PREPARE DATA
# ============================================================

reg_df = df[features + [target]].copy()

# CityId contains a few missing values
reg_df["CityId"] = reg_df["CityId"].fillna(-1)

# Remove remaining missing values
reg_df = reg_df.dropna()

print("\nRegression dataset shape:", reg_df.shape)

print("\nRating statistics:")
print(reg_df[target].describe())


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = reg_df[features]
y = reg_df[target]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. PREPROCESSING
# ============================================================

categorical_features = [
    "VisitMode",
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
# 7. DEFINE REGRESSION MODELS
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
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
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    mae = mean_absolute_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)

    results[model_name] = {
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2
    }

    trained_models[model_name] = pipeline

    print(f"RMSE     : {rmse:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"R2 Score : {r2:.4f}")


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results).T

# Higher R2 is better
results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n" + "=" * 60)
print("REGRESSION MODEL COMPARISON")
print("=" * 60)

print(results_df)


# ============================================================
# 10. SELECT BEST MODEL
# ============================================================

best_model_name = results_df["R2 Score"].idxmax()

best_model = trained_models[best_model_name]

print("\n" + "=" * 60)
print("BEST REGRESSION MODEL")
print("=" * 60)

print("Best model:", best_model_name)
print("R2 Score:", results_df.loc[best_model_name, "R2 Score"])
print("RMSE:", results_df.loc[best_model_name, "RMSE"])
print("MAE:", results_df.loc[best_model_name, "MAE"])


# ============================================================
# 11. SAVE BEST MODEL
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
    "rating_model.pkl"
)

joblib.dump(best_model, MODEL_PATH)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print("Saved to:", MODEL_PATH)

print("\nRegression completed successfully!")