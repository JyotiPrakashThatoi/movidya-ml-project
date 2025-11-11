#!/usr/bin/env python3
"""
train_learning_style_model.py

- Loads data/student-mat.csv (semicolon-separated)
- Derives 'learning_style' labels using simple explainable rules
- Trains DecisionTree (explainable) and RandomForest
- Performs GridSearchCV on RandomForest (production model)
- Compares results and saves:
    - explainable_model.joblib
    - production_model.joblib
    - learning_style_pipeline.joblib
    - learning_style_label_encoder.joblib
- Backups saved to ./artifacts/
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import argparse
import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# -----------------------
# Paths / output folders
# -----------------------
BASE = Path.cwd()
API_ARTIFACTS_DIR = BASE / "movidya-api" / "artifacts"
BACKUP_DIR = BASE / "artifacts"
API_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Label derivation (explainable rules)
# -----------------------
def derive_learning_style(row):
    """
    Simple rule-based derivation for learning style.
    - Visual: studytime >= 3 and failures <= 1
    - Auditory: goout >= 3
    - Kinesthetic: default
    Adjust rules as needed when you have labeled data.
    """
    try:
        studytime = int(row.get("studytime", 1))
        failures = int(row.get("failures", 0))
        goout = int(row.get("goout", 2))
        activities = str(row.get("activities", "no")).strip().lower()
        health = int(row.get("health", 3))
    except Exception:
        return "Kinesthetic"

    if studytime >= 3 and failures <= 1:
        return "Visual"
    elif goout >= 3:
        return "Auditory"
    else:
        return "Kinesthetic"

# -----------------------
# Load CSV
# -----------------------
def load_data(csv_path: str):
    df = pd.read_csv(csv_path, sep=";")
    return df

# -----------------------
# Preprocessor builder
# -----------------------
def build_preprocessor(numeric_features, categorical_features):
    """
    StandardScaler for numeric, OneHotEncoder for categorical.
    Uses sparse_output=False for recent sklearn versions.
    """
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop")
    return preprocessor

# -----------------------
# Main training flow
# -----------------------
def main(args):
    # Load
    df = load_data(args.data_path)
    print("Loaded dataset:", df.shape)

    # Candidate features (behavioral + lifestyle)
    candidate_features = [
        "age", "sex", "studytime", "failures", "goout",
        "activities", "health", "freetime", "absences"
    ]
    features = [c for c in candidate_features if c in df.columns]
    if not features:
        raise RuntimeError("No candidate features found in CSV. Check your CSV columns.")

    data = df[features].copy()
    data["learning_style"] = df.apply(derive_learning_style, axis=1)
    print("Label distribution:\n", data["learning_style"].value_counts())

    X = data[features].copy()
    y = data["learning_style"].copy()

    # Encode labels (required for consistent training and some algorithms)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)  # numeric labels
    # Save encoder for deployment
    encoder_path = API_ARTIFACTS_DIR / "learning_style_label_encoder.joblib"
    joblib.dump(le, encoder_path)
    joblib.dump(le, BACKUP_DIR / "learning_style_label_encoder.joblib")

    # Train/test split (use encoded labels for stratify)
    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, y_enc, test_size=0.20, stratify=y_enc, random_state=42
    )

    # Numeric vs categorical features
    numeric_features = [c for c in X_train.columns if c in ["age", "studytime", "failures", "goout", "health", "freetime", "absences"]]
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Models: DecisionTree (explainable) + RandomForest (candidate)
    dt_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", DecisionTreeClassifier(random_state=42))])
    rf_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))])

    trained_pipelines = {}
    metrics_summary = {}

    # Train DecisionTree
    print("\nTraining DecisionTree (explainable)...")
    dt_pipe.fit(X_train, y_train_enc)
    y_pred_dt_enc = dt_pipe.predict(X_test)
    y_test_labels = le.inverse_transform(y_test_enc)
    y_pred_dt_labels = le.inverse_transform(y_pred_dt_enc)
    dt_acc = accuracy_score(y_test_labels, y_pred_dt_labels)
    dt_f1 = f1_score(y_test_labels, y_pred_dt_labels, average="macro")
    trained_pipelines["DecisionTree"] = dt_pipe
    metrics_summary["DecisionTree"] = {"accuracy": dt_acc, "f1_macro": dt_f1}
    print(f"DecisionTree -> accuracy: {dt_acc:.4f}, f1_macro: {dt_f1:.4f}")
    print(classification_report(y_test_labels, y_pred_dt_labels, digits=4))

    # Train RandomForest baseline
    print("\nTraining RandomForest (baseline)...")
    rf_pipe.fit(X_train, y_train_enc)
    y_pred_rf_enc = rf_pipe.predict(X_test)
    y_pred_rf_labels = le.inverse_transform(y_pred_rf_enc)
    rf_acc = accuracy_score(y_test_labels, y_pred_rf_labels)
    rf_f1 = f1_score(y_test_labels, y_pred_rf_labels, average="macro")
    trained_pipelines["RandomForest"] = rf_pipe
    metrics_summary["RandomForest"] = {"accuracy": rf_acc, "f1_macro": rf_f1}
    print(f"RandomForest -> accuracy: {rf_acc:.4f}, f1_macro: {rf_f1:.4f}")
    print(classification_report(y_test_labels, y_pred_rf_labels, digits=4))

    # GridSearchCV on RandomForest for production candidate
    print("\nRunning GridSearchCV on RandomForest to refine hyperparameters...")
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 8, 16],
        "classifier__min_samples_split": [2, 5]
    }
    gs = GridSearchCV(rf_pipe, param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train_enc)
    gs_best = gs.best_estimator_
    y_pred_gs_enc = gs_best.predict(X_test)
    y_pred_gs_labels = le.inverse_transform(y_pred_gs_enc)
    gs_acc = accuracy_score(y_test_labels, y_pred_gs_labels)
    gs_f1 = f1_score(y_test_labels, y_pred_gs_labels, average="macro")
    print(f"GridSearch RandomForest -> accuracy={gs_acc:.4f}, f1_macro={gs_f1:.4f}")
    print("GridSearch best params:", gs.best_params_)

    # Comparison summary
    print("\nModel comparison summary:")
    for k, v in metrics_summary.items():
        print(f"- {k}: accuracy={v['accuracy']:.4f}, f1_macro={v['f1_macro']:.4f}")
    print(f"- RandomForest_GridSearch: accuracy={gs_acc:.4f}, f1_macro={gs_f1:.4f}")

    # Decision: production model = GridSearch RandomForest; explainable = DecisionTree
    production_pipeline = gs_best
    explainable_pipeline = trained_pipelines.get("DecisionTree")

    # Save artifacts
    explainable_path = API_ARTIFACTS_DIR / "explainable_model.joblib"
    production_path = API_ARTIFACTS_DIR / "production_model.joblib"
    pipeline_path = API_ARTIFACTS_DIR / "learning_style_pipeline.joblib"

    # Save explainable
    if explainable_pipeline is not None:
        joblib.dump(explainable_pipeline, explainable_path)
        joblib.dump(explainable_pipeline, BACKUP_DIR / "explainable_model.joblib")
        print(f"Saved explainable model (DecisionTree) to: {explainable_path}")

    # Save production model
    joblib.dump(production_pipeline, production_path)
    joblib.dump(production_pipeline, BACKUP_DIR / "production_model.joblib")
    print(f"Saved production model (GridSearch RandomForest) to: {production_path}")

    # Save preprocessor separately (optional, good to have)
    joblib.dump(preprocessor, pipeline_path)
    joblib.dump(preprocessor, BACKUP_DIR / "learning_style_pipeline.joblib")

    print("Saved encoder and pipeline backups to ./artifacts/")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Learning Style Classifier (RandomForest GridSearch for production)")
    parser.add_argument("--data_path", type=str, default="data/student-mat.csv", help="Path to student-mat.csv")
    args = parser.parse_args()
    main(args)
