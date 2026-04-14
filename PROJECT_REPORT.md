# FetalAI Project Report

## 1. Problem Understanding
Pregnancy monitoring depends on timely recognition of fetal distress and abnormal patterns in cardiotocography or related clinical measurements. Manual review can be slow, subjective, and difficult to scale for remote monitoring use cases. FetalAI addresses this by using supervised machine learning to classify fetal health from objective measurements.

## 2. Business Problem
The core business problem is to reduce the risk of delayed intervention during pregnancy by providing an automated decision-support system. The solution should help healthcare providers and patients identify potentially dangerous patterns early enough to improve care planning.

## 3. Business Requirements
The system should accept structured medical input, produce a fetal health prediction quickly, and be simple enough to deploy through a web application. It should also support model retraining, evaluation with multiple algorithms, and easy handoff to a Flask-based interface.

## 4. Literature Survey
Machine learning has been widely used in obstetric monitoring because it can learn complex relationships between maternal-fetal indicators and clinical outcomes. Classification models such as logistic regression, decision trees, random forests, support vector machines, and neural networks are commonly used for tabular health data. Balancing techniques like SMOTE are also frequently applied when the target classes are imbalanced.

## 5. Social and Business Impact
A reliable fetal health predictor can support early intervention, reduce unnecessary clinic visits through remote monitoring, and improve the quality of prenatal care. From a business perspective, it can lower operational burden for clinics while providing a differentiated digital-health service.

## 6. Data Collection and Preparation
The project uses the Kaggle fetal health classification dataset. The raw data is loaded from CSV, column names are normalized, and the target label is separated from the input features. Missing values and categorical variables are not present in this dataset, so the main preparation step is handling class imbalance.

## 7. Exploratory Data Analysis
EDA includes descriptive statistics, histograms, scatter plots, and correlation analysis. Descriptive statistics help identify the range and spread of each feature. Visual analysis helps show class separation and feature distribution. Correlation analysis helps identify the strongest contributors to fetal health classification.

## 8. Model Building
Multiple algorithms are trained and compared, including logistic regression, k-nearest neighbors, decision tree, random forest, gradient boosting, support vector machine, and MLP classifier. The training data is balanced using SMOTE, and the features are standardized before fitting each classifier. The best model is selected using macro F1 score.

## 9. Testing and Performance Evaluation
The trained models are evaluated on the test split using accuracy, precision, recall, macro F1 score, confusion matrix, and a classification report. These metrics provide a more complete view than accuracy alone, especially for imbalanced health data.

## 10. Model Deployment
The best-performing pipeline is saved as `flask/model.pkl`. The deployment layer loads the saved model, accepts feature values through a form, converts the inputs into a prediction-ready dataframe, and displays the predicted class in the browser.

## 11. Project Flow
The end-to-end flow is: user input is entered through the UI, the Flask app converts the input into model features, the saved pipeline returns a fetal health class, and the result is displayed immediately on the web page.

## 12. Project Structure
- `Training/train_model.py` trains and exports the model.
- `flask/app.py` serves the prediction UI and loads the saved model.
- `flask/templates/` stores the HTML pages.
- `flask/static/css/style.css` contains the interface styling.
- `data/fetal_health.csv` stores the training dataset.

## 13. Project Demonstration and Documentation
For a project demonstration, show the dataset loading step, feature inspection, model comparison, saved model creation, and Flask prediction flow. The documentation should include the problem definition, data preparation, EDA summary, model comparison results, deployment architecture, and prediction screenshots.

## 14. Conclusion
FetalAI combines tabular machine learning with a Flask deployment layer to provide an accessible fetal health prediction workflow. The project is designed to support early intervention, remote monitoring, and risk prediction in a practical healthcare setting.
