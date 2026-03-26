import pandas as pd
import shap


def prepare_xgb_explanation_data(xgb_pipeline, X_test, numeric_features, categorical_features):
    preprocessor_fitted = xgb_pipeline.named_steps['preprocessor']
    xgb_model = xgb_pipeline.named_steps['classifier']

    cat_encoder = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot']
    cat_features_encoded = cat_encoder.get_feature_names_out(categorical_features)
    all_feature_names = list(numeric_features) + list(cat_features_encoded)

    X_test_transformed = preprocessor_fitted.transform(X_test)
    X_test_df = pd.DataFrame(X_test_transformed, columns=all_feature_names)

    X_test_df.columns = [
        col.replace('[', '_').replace(']', '_').replace('<', '_')
        for col in X_test_df.columns
    ]

    return xgb_model, X_test_df


def run_shap_explanations(xgb_model, X_test_df, applicant_idx=0):
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test_df)

    print("\n--- GLOBAL EXPLANATION (Portfolio Level) ---")
    shap.summary_plot(shap_values, X_test_df)

    print(f"\n--- LOCAL EXPLANATION (Applicant #{applicant_idx}) ---")
    shap.plots.waterfall(shap_values[applicant_idx])
