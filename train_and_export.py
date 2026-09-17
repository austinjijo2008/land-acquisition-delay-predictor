"""
Export script for the Land Acquisition Delay Prediction model.

This reproduces the original Colab notebook's cleaning, feature engineering,
train/test split, and model choices (original_notebook.ipynb), with one
deliberate fix over the notebook's own cell order:

  ORIGINAL NOTEBOOK BUG: cell 15 scales numeric_columns on the full dataframe
  BEFORE the train/test split. Cell 36 then fits a SECOND scaler ("new_scaler")
  on X_train's numeric_features -- but those columns were already scaled by
  cell 15, so new_scaler ends up fit on already-standardized data (mean~=0,
  scale~=1), making it nearly a no-op. Since predict_with_user_input (cell 42)
  applies only new_scaler to RAW user input, raw values pass through almost
  unchanged and dominate the logistic regression's logit, causing the deployed
  model to predict "Delayed" at ~100% confidence for nearly any input,
  regardless of actual risk profile (confirmed by testing: a clearly low-risk
  and a clearly high-risk profile both came back "Delayed, ~100%").

  THE FIX (approved by the project owner after this was diagnosed): remove the
  redundant first full-dataframe scaling pass. new_scaler becomes the ONLY
  scaling step, fit on X_train's raw numeric columns, and every model
  (delay_model, decision tree, random forest, final_model) is trained on data
  that has gone through that exact same single scaling pass -- matching what
  predict_with_user_input assumes happens at inference time. Nothing else
  changes: same algorithms, same hyperparameters, same features, same
  train/test split, same random_state=42.

  cells 1-13   -> data cleaning (unchanged)
  cell 17      -> one-hot encoding (unchanged)
  cells 18-20  -> train/test split (unchanged)
  (fix)        -> single StandardScaler fit on X_train's raw numeric columns
  cell 21-29   -> Logistic Regression / Decision Tree / Random Forest, now
                  trained on the singly-scaled X_train
  cell 31      -> final_model = LogisticRegression(max_iter=1000)  [DEPLOYED MODEL]

Run this once (locally or in Colab) whenever you have a new/updated
land_acquisition_delay_synthetic.csv, to regenerate the model/ artifacts
used by the Streamlit app. It does not retrain anything the app itself runs.

Usage:
    python train_and_export.py --csv data/land_acquisition_delay_synthetic.csv
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from ml_backend import (
    BACKEND,
    train_test_split,
    StandardScaler,
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def main(csv_path: Path, out_dir: Path):
    # ---- cells 1-13: load + clean (same order as the notebook) ----
    df = pd.read_csv(csv_path)

    df["Land_Area_Acres"] = df["Land_Area_Acres"].fillna(df["Land_Area_Acres"].median())
    df["Compensation_Dispute"] = df["Compensation_Dispute"].fillna(df["Compensation_Dispute"].mode()[0])

    df = df.drop_duplicates()

    df["Number_of_Owners"] = df["Number_of_Owners"].fillna(df["Number_of_Owners"].median())
    df["Legal_Cases"] = df["Legal_Cases"].fillna(df["Legal_Cases"].median())
    df["Documents_Missing"] = df["Documents_Missing"].fillna(df["Documents_Missing"].median())
    df["Approval_Days"] = df["Approval_Days"].fillna(df["Approval_Days"].median())

    # Save raw (pre-scaling) numeric stats for building sensible Streamlit input ranges.
    raw_numeric_for_ui = [
        "Land_Area_Acres", "Number_of_Owners", "Number_of_Parcels", "Legal_Cases",
        "Ownership_Disputes", "Documents_Missing", "Objections_Received",
        "Affected_Families", "Government_Clearances", "Planned_Duration_Days",
        "Approval_Days",
    ]
    numeric_stats = {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
        }
        for col in raw_numeric_for_ui
    }

    # Save raw category options + the base category get_dummies(drop_first=True) will drop
    # (alphabetically first), derived from the actual data rather than hardcoded.
    categorical_columns = [
        "Project_Type", "Location_Zone", "Land_Type",
        "Rehabilitation_Required", "Compensation_Dispute", "Previous_Acquisition_Delay",
    ]
    categorical_options = {}
    for col in categorical_columns:
        options = sorted(df[col].dropna().unique().tolist())
        categorical_options[col] = {
            "options": options,
            "base_category": options[0],  # what drop_first=True drops
        }

    # ---- cell 17: one-hot encode (unchanged) ----
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # ---- cell 18-20: split (unchanged) ----
    X = df.drop(["Delayed", "Actual_Duration_Days", "Project_ID"], axis=1, errors="ignore")
    y = df["Delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- FIX: single scaling pass, fit on X_train's RAW numeric columns ----
    # This is the only scaler in the pipeline now (see module docstring for why
    # the notebook's original double-scaling made the deployed model unusable).
    numeric_features = [
        "Land_Area_Acres", "Number_of_Owners", "Number_of_Parcels", "Legal_Cases",
        "Ownership_Disputes", "Documents_Missing", "Objections_Received",
        "Affected_Families", "Government_Clearances", "Planned_Duration_Days",
        "Approval_Days",
    ]
    new_scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numeric_features] = new_scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = new_scaler.transform(X_test[numeric_features])

    metrics = {}

    # ---- cell 21-25: Logistic Regression (delay_model) ----
    delay_model = LogisticRegression()
    delay_model.fit(X_train, y_train)
    pred = delay_model.predict(X_test)
    metrics["logistic_regression"] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }

    # ---- cell 26: Decision Tree ----
    dt_model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    metrics["decision_tree"] = {
        "accuracy": float(accuracy_score(y_test, dt_pred)),
        "confusion_matrix": confusion_matrix(y_test, dt_pred).tolist(),
    }

    # ---- cell 27-29: Random Forest + feature importance ----
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    metrics["random_forest"] = {
        "accuracy": float(accuracy_score(y_test, rf_pred)),
        "confusion_matrix": confusion_matrix(y_test, rf_pred).tolist(),
        "classification_report": classification_report(y_test, rf_pred, output_dict=True),
    }
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": rf_model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)
    metrics["random_forest"]["feature_importance"] = importance_df.to_dict(orient="records")

    # ---- cell 30: the notebook's own hardcoded comparison chart values ----
    metrics["notebook_comparison_chart"] = {
        "models": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "accuracy": [0.601, 0.553, 0.582],
    }

    # ---- cell 31: final_model (the deployed model), trained on the singly-scaled X_train ----
    final_model = LogisticRegression(max_iter=1000)
    final_model.fit(X_train, y_train)

    # Deployed model's own accuracy, evaluated the same way cell 23 evaluates delay_model
    # (accuracy_score(y_test, model.predict(X_test))), applied to the actual deployed
    # model + scaler pipeline. Not present as a printed cell in the notebook, but computed
    # here using the notebook's own imported evaluation function against the real pipeline.
    final_pred = final_model.predict(X_test)
    metrics["deployed_final_model"] = {
        "algorithm": "LogisticRegression(max_iter=1000)",
        "accuracy": float(accuracy_score(y_test, final_pred)),
        "confusion_matrix": confusion_matrix(y_test, final_pred).tolist(),
        "classification_report": classification_report(y_test, final_pred, output_dict=True),
        "note": (
            "Computed by this export script using the same evaluation method as the "
            "notebook (accuracy_score/confusion_matrix/classification_report), applied "
            "to the actual deployed final_model + new_scaler pipeline. Not a printed "
            "cell output in the original notebook."
        ),
    }

    # ---- save artifacts ----
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "final_model.pkl", "wb") as f:
        pickle.dump(final_model, f)

    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(new_scaler, f)

    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(list(X_train.columns), f, indent=2)

    with open(out_dir / "numeric_features.json", "w") as f:
        json.dump(numeric_features, f, indent=2)

    with open(out_dir / "categorical_options.json", "w") as f:
        json.dump(categorical_options, f, indent=2)

    with open(out_dir / "numeric_stats.json", "w") as f:
        json.dump(numeric_stats, f, indent=2)

    metrics["_ml_backend"] = BACKEND

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"ML backend used: {BACKEND}")
    print(f"Artifacts written to {out_dir.resolve()}")
    print(f"Deployed model (final_model) test accuracy: {metrics['deployed_final_model']['accuracy']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=Path("data/land_acquisition_delay_synthetic.csv"),
        help="Path to land_acquisition_delay_synthetic.csv",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("model"),
        help="Output directory for exported artifacts",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(
            f"CSV not found at {args.csv}. Export it from Colab first, e.g.:\n"
            f"  from google.colab import files\n"
            f"  files.download('/content/land_acquisition_delay_synthetic.csv')\n"
            f"then place it at {args.csv} (or pass --csv <path>)."
        )

    main(args.csv, args.out)
