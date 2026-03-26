from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


def compute_scale_pos_weight(y_train):
    return (y_train == 0).sum() / (y_train == 1).sum()


def build_models(scale_pos_weight):
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced', max_depth=5),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42
        )
    }
    return models


def train_and_evaluate_models(models, preprocessor, X_train, X_test, y_train, y_test):
    trained_models = {}

    for name, model in models.items():
        clf = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        print(f"{name} ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
        trained_models[name] = clf

    return trained_models
