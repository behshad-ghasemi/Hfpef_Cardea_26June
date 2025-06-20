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
import joblib

import sklearn
print(sklearn.__version__)

FEATURES = [
    'Epicardial fat thickness (mm)', 'LV mass i (g/m2) Calcolo automatico',
    'LAD (mm)','LVEF (%)','E/e avg', 'DM','WHO-FC','PAS (mmHg)', 'NT-pro-BNP (pg/mL)', 'AF']

NUM_FEATURES = [
    'Epicardial fat thickness (mm)',
    'LV mass i (g/m2) Calcolo automatico', 'LAD (mm)','LVEF (%)','E/e avg', 'PAS (mmHg)', 'NT-pro-BNP (pg/mL)'
]

CAT_FEATURES = ['DM', 'WHO-FC', 'AF']

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
    if feature == "AF":
        user_input[feature] = st.selectbox("AF (Atrial Fibrillation):", options=["0", "1"])
    elif feature == "DM":
        user_input[feature] = st.selectbox("DM (Diabetes Mellitus):", options=["0", "1"])
    elif feature == "WHO-FC":
        user_input[feature] = st.selectbox("WHO-FC (Functional Class):", options=["1", "2", "3", "4"])
    elif any(x in feature.lower() for x in ["assente", "presente", "no", "si", "0", "1", "2"]):
        user_input[feature] = st.number_input(f"{feature}:", step=1.0)
    else:
        user_input[feature] = st.number_input(f"{feature}:", step=0.1)

if st.button("🔍 Estimate 🔍"):
    try:
        input_df = pd.DataFrame([user_input])
        
       
        for cat_feature in CAT_FEATURES:
            if cat_feature in input_df.columns:
                input_df[cat_feature] = input_df[cat_feature].astype('category')

        transformed_input = pipeline.transform(input_df)

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

        # نمودار مقایسه اول
        fig, ax = plt.subplots()
        sns.barplot(x=["Logistic", "Random Forest", "XGBoost"], y=[prob_log, prob_rf, prob_gb], palette="Set2", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Predicted Probability")
        ax.set_title("Model Comparison")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ {e}")
        st.success("💃🥳YOHOOOOOOOOOO, Low Risk of HFpEF 🥳💃")


    

        

        
    except Exception as feature_error:
        st.warning(f"Feature importance analysis could not be performed: {feature_error}")
        st.error(f"Details: {str(feature_error)}")
