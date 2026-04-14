# FetalAI

FetalAI is an end-to-end machine learning web application for fetal health risk classification.
It combines a multi-model training pipeline with a Flask deployment layer to predict fetal health status as:

- Normal
- Suspect
- Pathological

The project is designed for academic demonstration and practical screening workflows such as early intervention support, remote monitoring, and risk prioritization.

## Key Highlights

- End-to-end ML workflow: preprocessing, model training, evaluation, and deployment
- Multiple classifier comparison: Random Forest, Decision Tree, Logistic Regression, KNN
- Imbalance handling with SMOTE
- Multi-page professional Flask UI (Home, About, Contact, Inspect, Output)
- Glassmorphism and skeuomorphic interface styling with responsive behavior
- Saved production artifact (`model.pkl`) reused at inference time

## Current Best Model

Based on the latest training run:

- Best model: Random Forest Classifier
- Accuracy: 0.9279
- Precision (macro): 0.8657
- Recall (macro): 0.8843
- F1 (macro): 0.8744

Metrics source files:

- `Training/training_report.json`
- `flask/feature_metadata.json`

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- Flask (web framework)
- HTML, CSS (custom UI)

## Project Structure

```text
fetalAI/
  README.md
  requirements.txt
  data/
    fetal_health.csv
  Training/
    train_model.py
    training_report.json
    model_comparison.png
  flask/
    app.py
    model.pkl
    feature_metadata.json
    static/
      css/
        style.css
      images/
        logo.png
        heroimage.png
        fetalai-logo.svg
        home-hero-bg.svg
    templates/
      base.html
      index.html
      about.html
      contact.html
      inspect.html
      outputt.html
```

## Dataset

- Source: Kaggle - Fetal Health Classification
- Link: https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification
- Required local path: `data/fetal_health.csv`

## Training Pipeline

Training is handled by `Training/train_model.py`.

Pipeline flow:

1. Load CSV and normalize column names
2. Validate required features and target (`fetal_health`)
3. Select inference feature set (8 model input features)
4. Split into train/test (stratified)
5. Apply SMOTE on training split
6. Train four classifiers
7. Evaluate using accuracy, precision, recall, F1, confusion matrix
8. Compare models and select best by accuracy
9. Save best model and metadata for Flask inference

Generated artifacts:

- `flask/model.pkl`
- `flask/feature_metadata.json`
- `Training/training_report.json`
- `Training/model_comparison.png`

## Web Application Flow

Flask routes:

- `/` -> Home
- `/about` -> Domain and project context
- `/contact` -> Contact form with mandatory fields and success popup
- `/inspect` -> Input form for prediction features
- `/predict` -> POST inference endpoint

Prediction flow:

1. User enters values on Inspect page
2. Flask builds a dataframe in the expected feature order
3. Saved model predicts class and confidence (if available)
4. Result is rendered on Output page

## Setup and Run

1. Create and activate virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Place dataset file

```text
data/fetal_health.csv
```

4. Train model

```bash
python Training/train_model.py
```

5. Run Flask app

```bash
python flask/app.py
```

6. Open in browser

```text
http://127.0.0.1:5000
```

## Validation Checklist

- `model.pkl` exists in `flask/`
- `feature_metadata.json` contains `best_model`
- `training_report.json` contains model comparison metrics
- Inspect form accepts inputs and returns prediction on Output page
- Contact form enforces mandatory fields and shows 3-second success popup

## Notes

- This application is a decision-support prototype and not a standalone clinical diagnosis system.
- If dataset or features are changed, retrain the model before running the web app.
- Keep `train_model.py` and `app.py` feature order aligned for correct inference behavior.
