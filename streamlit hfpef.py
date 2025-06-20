import streamlit as st
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import shap

import sklearn
print(sklearn.__version__)

FEATURES = [
    'Epicardial fat thickness (mm)', 'LV mass i (g/m2) Calcolo automatico',
    'LAD (mm)','LVEF (%)','E/e avg', 'DM','WHO-FC','PAS (mmHg)', 'NT-pro-BNP (pg/mL)', 'AF']

NUM_FEATURES = [
    'Epicardial fat thickness (mm)',
    'LV mass i (g/m2) Calcolo automatico', 'LAD (mm)','LVEF (%)','E/e avg', 'PAS (mmHg)', 'NT-pro-BNP (pg/mL)'
]

CAT_FEATURES = ['DM', 'WHO-FC', 'FA']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), NUM_FEATURES),

        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), CAT_FEATURES)
    ],
    remainder='drop'
)

full_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('pca', PCA(n_components=0.95))
])

st.title("🫀 HFpEF Probability 🫀")

st.markdown("Do not hesitate to reach us for further questions:")
st.markdown(" 🏢 Dr. Fusco , Manager: info@cardeasrl.it ")
st.markdown(" 👩‍💻 Behshad , programmer : b.ghaseminezhadabdol@studio.unibo.it ")
st.markdown(" ")
st.markdown("  ")
st.markdown("Insert the patient's clinical data below, to detect the probability of having HFpEF. ")

@st.cache_resource
def load_models():
    try:
        pipeline = joblib.load("data_with_pca.joblib")
        log_model = joblib.load("Logistic_model.joblib")
        rf_model = joblib.load("randomforest_model.joblib")
        xgb_model = joblib.load("xgboost_model.joblib")
        return pipeline, log_model, rf_model, xgb_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

pipeline, log_model, rf_model, xgb_model = load_models()

user_input = {}
for feature in FEATURES:
    user_input[feature] = st.number_input(f"{feature}:", step=0.1)

if st.button("🔍 Estimate 🔍"):
    try:
        input_df = pd.DataFrame([user_input])
        transformed_input = pipeline.transform(input_df)

        # Get feature names BEFORE PCA
        features_before_pca = pipeline.named_steps['preprocessing'].get_feature_names_out()
        df_for_shap = pd.DataFrame(pipeline.named_steps['preprocessing'].transform(input_df), columns=features_before_pca)

        explainer = shap.Explainer(xgb_model)
        shap_values = explainer(df_for_shap)

        st.subheader("🎯 SHAP Explanation for this Patient (XGBoost)")
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)

        prob_log = log_model.predict_proba(transformed_input)[0][1]
        prob_rf = rf_model.predict_proba(transformed_input)[0][1]
        prob_gb = xgb_model.predict_proba(transformed_input)[0][1]

        st.subheader("🤔 Prediction Probabilities")
        st.write(f"🔹 **Logistic Regression**: `{prob_log:.4f}`")
        st.write(f"🔹 **Random Forest**: `{prob_rf:.4f}`")
        st.write(f"🔹 **XG Boosting**: `{prob_gb:.4f}`")

        if prob_gb > 0.6:
            st.error("🚨💀 High Risk of HFpEF Detected! 😱")
        else:
            st.success("✅ Low Risk of HFpEF Detected 🎉")

        fig, ax = plt.subplots(figsize=(6, 5))
        models = ["Logistic Regression", "Random Forest", "XG Boosting"]
        probabilities = [prob_log, prob_rf, prob_gb]
        sns.barplot(x=models, y=probabilities, palette='mako', ax=ax)
        ax.set_title("Model Probability Comparison  ")
        ax.set_ylabel("HFpEF Probability ")
        st.pyplot(fig)

        def plot_feature_importance(model):
            importance = model.feature_importances_
            features = FEATURES  # Make sure order matches
            importance_df = pd.DataFrame({"Feature": features, "Importance": importance})
            importance_df = importance_df.sort_values(by="Importance", ascending=False)

            st.write("### Feature Importance - Random Forest")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
            st.pyplot(fig)
            return fig

        feature_importance_fig = plot_feature_importance(xgb_model)

    except Exception as e:
        st.error(f"❌ {e}")
