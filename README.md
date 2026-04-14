# FetalAI

FetalAI is a Flask-based machine learning project for predicting fetal health from clinical measurements. It supports the full workflow described in the project brief: data preparation, exploratory data analysis, model training, model evaluation, and deployment through a web UI.

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
  flask/
    app.py
    model.pkl
    feature_metadata.json
    static/
      css/
        style.css
    templates/
      base.html
      index.html
      inspect.html
      outputt.html
```

## Setup

1. Create and activate a Python environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Download the Kaggle dataset and place it at `data/fetal_health.csv`.
4. Train the model by running `python Training/train_model.py`.
5. Start the Flask app with `python flask/app.py`.

## Notes

- The training script normalizes dataset column names so the Flask app can use snake_case fields consistently.
- The training workflow compares Random Forest, Decision Tree, Logistic Regression, and KNN.
- The best model is selected by accuracy and exported for Flask inference.
- The saved model is written to `flask/model.pkl`.
- The app reads `flask/feature_metadata.json` when available to keep the deployment metadata in sync with the trained model.
- Web flow is split into pages: `index.html` (landing), `inspect.html` (input form), and `outputt.html` (prediction result).
- If you want to regenerate the model after updating the dataset, rerun the training script before launching the app.
