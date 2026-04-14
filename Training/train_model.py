"""Train and export the FetalAI classification model.

Expected dataset location:
    data/fetal_health.csv

The script follows the project flow by training multiple algorithms
(Random Forest, Decision Tree, Logistic Regression, KNN), evaluating them
with multiple metrics, comparing performance, and saving the best model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "fetal_health.csv"
FLASK_DIR = ROOT_DIR / "flask"
MODEL_PATH = FLASK_DIR / "model.pkl"
METADATA_PATH = FLASK_DIR / "feature_metadata.json"
REPORT_PATH = ROOT_DIR / "Training" / "training_report.json"
TARGET_COLUMN = "fetal_health"

EXPECTED_FEATURES = [
    "baseline_value",
    "accelerations",
    "fetal_movement",
    "uterine_contractions",
    "light_decelerations",
    "severe_decelerations",
    "prolongued_decelerations",
    "abnormal_short_term_variability",
    "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability",
    "histogram_width",
    "histogram_min",
    "histogram_max",
    "histogram_number_of_peaks",
    "histogram_number_of_zeroes",
    "histogram_mode",
    "histogram_mean",
    "histogram_median",
    "histogram_variance",
    "histogram_tendency",
]

# Feature subset used in the original notebook and Flask input form.
SELECTED_FEATURES = [
    "prolongued_decelerations",
    "abnormal_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "histogram_variance",
    "histogram_median",
    "mean_value_of_long_term_variability",
    "histogram_mode",
    "accelerations",
]

LABELS = {
    1: "Normal",
    2: "Suspect",
    3: "Pathological",
}


def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    renamed = dataframe.copy()
    renamed.columns = (
        renamed.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return renamed


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the Kaggle dataset and place it at data/fetal_health.csv."
        )
    return pd.read_csv(path)


def evaluate_model(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }


def random_forest_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(x_train, y_train)
    return {
        "name": "Random Forest Classifier",
        "estimator": model,
        "metrics": evaluate_model(model, x_test, y_test),
    }


def decision_tree_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    model = DecisionTreeClassifier(random_state=42)
    model.fit(x_train, y_train)
    return {
        "name": "Decision Tree Classifier",
        "estimator": model,
        "metrics": evaluate_model(model, x_test, y_test),
    }


def logistic_regression_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    return {
        "name": "Logistic Regression",
        "estimator": model,
        "metrics": evaluate_model(model, x_test, y_test),
    }


def knn_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]
    )
    model.fit(x_train, y_train)
    return {
        "name": "K Neighbors Classifier",
        "estimator": model,
        "metrics": evaluate_model(model, x_test, y_test),
    }


def compare_models(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        rows.append(
            {
                "name": result["name"],
                "accuracy": result["metrics"]["accuracy"],
                "precision_macro": result["metrics"]["precision_macro"],
                "recall_macro": result["metrics"]["recall_macro"],
                "f1_macro": result["metrics"]["f1_macro"],
            }
        )
    rows.sort(key=lambda row: row["accuracy"], reverse=True)
    return rows


def save_comparison_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    names = [row["name"] for row in rows]
    scores = [row["accuracy"] for row in rows]

    plt.figure(figsize=(10, 5))
    plt.barh(names, scores)
    plt.xlabel("accuracy")
    plt.ylabel("name")
    plt.title("Model Comparison by Accuracy")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    dataframe = normalize_columns(load_dataset(DATA_PATH))

    if TARGET_COLUMN not in dataframe.columns:
        raise KeyError(f"Expected target column '{TARGET_COLUMN}' was not found in the dataset.")

    feature_columns = [column for column in dataframe.columns if column != TARGET_COLUMN]
    missing_expected = [column for column in EXPECTED_FEATURES if column not in feature_columns]
    if missing_expected:
        raise KeyError(f"The dataset is missing expected feature columns: {missing_expected}")

    x = dataframe[SELECTED_FEATURES]
    y = dataframe[TARGET_COLUMN].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    smote = SMOTE(random_state=42)
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

    training_results = [
        random_forest_model(x_train_smote, x_test, y_train_smote, y_test),
        decision_tree_model(x_train_smote, x_test, y_train_smote, y_test),
        logistic_regression_model(x_train_smote, x_test, y_train_smote, y_test),
        knn_model(x_train_smote, x_test, y_train_smote, y_test),
    ]

    comparison_rows = compare_models(training_results)
    best_name = comparison_rows[0]["name"]
    best_result = next(result for result in training_results if result["name"] == best_name)
    best_model = best_result["estimator"]

    FLASK_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    comparison_plot = ROOT_DIR / "Training" / "model_comparison.png"
    save_comparison_plot(comparison_rows, comparison_plot)

    metadata = {
        "feature_names": x.columns.tolist(),
        "target_name": TARGET_COLUMN,
        "labels": LABELS,
        "best_model": best_name,
        "best_score": best_result["metrics"]["accuracy"],
        "metrics": best_result["metrics"],
    }

    report_payload = {
        "comparison": comparison_rows,
        "details": [
            {
                "model": result["name"],
                **result["metrics"],
            }
            for result in training_results
        ],
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print(f"Saved best model to: {MODEL_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")
    print(f"Saved comparison plot to: {comparison_plot}")
    print(f"Best model: {best_name}")
    print(f"Best accuracy: {best_result['metrics']['accuracy']:.4f}")
    print("Evaluation summary:")
    for row in comparison_rows:
        print(
            f"- {row['name']}: accuracy={row['accuracy']:.4f}, precision={row['precision_macro']:.4f}, "
            f"recall={row['recall_macro']:.4f}, f1={row['f1_macro']:.4f}"
        )


if __name__ == "__main__":
    main()
