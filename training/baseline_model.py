from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Пути

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hr_attrition.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "baseline_model.joblib"

# Загрузка данных

df = pd.read_csv(DATA_PATH)

# Target

df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

# Удаляем мусор

drop_columns = [
    "EmployeeCount",
    "Over18",
    "StandardHours",
    "EmployeeNumber"
]

df = df.drop(columns=drop_columns)

# X / y

X = df.drop(columns=["Attrition"])
y = df["Attrition"]

# Категориальные признаки

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

# Препроцессинг

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

# Модель

model = LogisticRegression(
    max_iter=3000,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train

pipeline.fit(X_train, y_train)

# Evaluate

y_pred = pipeline.predict(X_test)

y_proba = pipeline.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_proba)

f1 = f1_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

print("\n===== BASELINE MODEL =====")

print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# Save model

joblib.dump(
    pipeline,
    MODEL_PATH
)

print(f"\nModel saved: {MODEL_PATH}")

__all__ = ['pipeline', 'X_test', 'y_test', 'roc_auc', 'f1', 'precision', 'recall']