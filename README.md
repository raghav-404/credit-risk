# Credit Risk Modeling

A modular Python project for credit risk prediction using XGBoost and SHAP-based explainability.

## Dataset

German Credit dataset (credit-g) from OpenML. Target: 0 = Good Loan, 1 = Default.

## Models

- Logistic Regression
- Decision Tree
- XGBoost

Evaluation metric: ROC-AUC Score

## Explainability

- SHAP summary plot (global)
- SHAP waterfall plot (local, per applicant)

## Tech Stack

Python
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
uv
