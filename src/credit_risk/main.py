import warnings
warnings.filterwarnings('ignore')

from credit_risk.config import APPLICANT_IDX
from credit_risk.data_loader import load_data
from credit_risk.eda import plot_class_distribution, plot_feature_correlations
from credit_risk.preprocessing import split_data, build_preprocessor
from credit_risk.modeling import compute_scale_pos_weight, build_models, train_and_evaluate_models
from credit_risk.explain import prepare_xgb_explanation_data, run_shap_explanations


def main():
    df = load_data()

    plot_class_distribution(df)
    plot_feature_correlations(df)

    X, y, X_train, X_test, y_train, y_test = split_data(df)
    preprocessor, numeric_features, categorical_features = build_preprocessor(X)

    scale_pos_weight = compute_scale_pos_weight(y_train)
    models = build_models(scale_pos_weight)

    trained_models = train_and_evaluate_models(
        models=models,
        preprocessor=preprocessor,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    xgb_pipeline = trained_models["XGBoost"]

    xgb_model, X_test_df = prepare_xgb_explanation_data(
        xgb_pipeline=xgb_pipeline,
        X_test=X_test,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    run_shap_explanations(
        xgb_model=xgb_model,
        X_test_df=X_test_df,
        applicant_idx=APPLICANT_IDX
    )


if __name__ == "__main__":
    main()
