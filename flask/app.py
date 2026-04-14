from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
METADATA_PATH = BASE_DIR / "feature_metadata.json"

app = Flask(__name__)
app.secret_key = "fetalai-secret-key"

FIELD_GROUPS = [
    {
        "title": "Fetal Health Inputs",
        "fields": [
            {"name": "prolongued_decelerations", "label": "Prolongued Decelerations", "default": 0.0, "step": "any"},
            {"name": "abnormal_short_term_variability", "label": "Abnormal Short Term Variability", "default": 47, "step": "any"},
            {"name": "percentage_of_time_with_abnormal_long_term_variability", "label": "Percentage of Time With Abnormal Long Term Variability", "default": 9, "step": "any"},
            {"name": "histogram_variance", "label": "Histogram Variance", "default": 17, "step": "any"},
            {"name": "histogram_median", "label": "Histogram Median", "default": 137, "step": "any"},
            {"name": "mean_value_of_long_term_variability", "label": "Mean Value of Long Term Variability", "default": 8, "step": "any"},
            {"name": "histogram_mode", "label": "Histogram Mode", "default": 141, "step": "any"},
            {"name": "accelerations", "label": "Accelerations", "default": 0.003, "step": "any"},
        ],
    },
]

LABELS = {
    1: "Normal",
    2: "Suspect",
    3: "Pathological",
}


def load_model() -> Any | None:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def load_metadata() -> dict[str, Any]:
    if METADATA_PATH.exists():
        try:
            return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {
        "best_model": "Not trained yet",
        "labels": LABELS,
        "feature_names": [field["name"] for group in FIELD_GROUPS for field in group["fields"]],
    }


def flatten_fields() -> list[dict[str, Any]]:
    return [field for group in FIELD_GROUPS for field in group["fields"]]


def build_input_frame(form_data: dict[str, str]) -> pd.DataFrame:
    values = []
    for field in flatten_fields():
        raw_value = form_data.get(field["name"], field["default"])
        values.append(float(raw_value))
    return pd.DataFrame([values], columns=[field["name"] for field in flatten_fields()])


def prediction_label(raw_prediction: Any) -> str:
    try:
        numeric_prediction = int(raw_prediction)
    except (TypeError, ValueError):
        return str(raw_prediction)
    return LABELS.get(numeric_prediction, f"Class {numeric_prediction}")


@app.route("/")
def index() -> str:
    metadata = load_metadata()
    model_ready = load_model() is not None
    return render_template(
        "index.html",
        metadata=metadata,
        model_ready=model_ready,
    )


@app.route("/inspect")
def inspect() -> str:
    metadata = load_metadata()
    model_ready = load_model() is not None
    return render_template(
        "inspect.html",
        field_groups=FIELD_GROUPS,
        metadata=metadata,
        model_ready=model_ready,
    )


@app.route("/predict", methods=["POST"])
def predict() -> str:
    metadata = load_metadata()
    model = load_model()
    model_ready = model is not None

    if not model_ready:
        flash(
            "Saved model not found. Run Training/train_model.py after downloading the Kaggle dataset.",
            "error",
        )
        return render_template(
            "inspect.html",
            field_groups=FIELD_GROUPS,
            metadata=metadata,
            model_ready=model_ready,
        )

    try:
        input_frame = build_input_frame(request.form)
        prediction = model.predict(input_frame)[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_frame)[0]
            confidence = float(np.max(probabilities))

        input_values = {
            field["name"]: float(request.form.get(field["name"], field["default"]))
            for field in flatten_fields()
        }
        return render_template(
            "outputt.html",
            field_groups=FIELD_GROUPS,
            metadata=metadata,
            prediction_code=int(prediction),
            prediction_text=prediction_label(prediction),
            confidence=confidence,
            input_values=input_values,
            model_ready=model_ready,
        )
    except ValueError:
        flash("Please enter valid numeric values for all fields.", "error")
        return render_template(
            "inspect.html",
            field_groups=FIELD_GROUPS,
            metadata=metadata,
            model_ready=model_ready,
        )


@app.context_processor
def inject_helpers() -> dict[str, Any]:
    return {
        "page_title": "FetalAI",
        "field_count": len(flatten_fields()),
    }


if __name__ == "__main__":
    app.run(debug=True)
