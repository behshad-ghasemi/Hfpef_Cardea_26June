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

FEATURES = ["BSA (non toccare)", 'Epicardial fat thickness (mm)', 'LVEDD (mm)', 'LVESD (mm)', 'LV mass (g)',
            'LV mass i (g/m2) Calcolo automatico', 'LAD (mm)', 'LAV (ml)', 'iLAV (mL/m2) Calcolo automatico',
            'LVEF (%)', 'MR 0=no, 1=min, 2=lieve, 3=mod, 4=sev', 'E/A', 'E/e avg', 'VCI diameter (mm)',
            'sPAP (mmHg) = RV-RA + RAP (3-8-15)', 'Probabilità IP a riposo (1, bassa; 2, intermedia; 3, alta)', 'eta',
            'BSA', 'IAS (0, assente; 1, presente)', 'Dislipidemia (0, assente; 1, presente, 2 s. metabolica)',
            'Diabete mellito (0, assente; 1, presente, 2 diabete complicato)', 'Beta bloccante 1 si, 0 no',
            'MRA 1 si, 0 no', 'ACE/ARB 1 si, 0 no',
            'Terapia antiscompenso assente (0), sub-ottimale (1-2), ottimale (3)',
            "Insufficienza renale lieve (si 1; no 0)", "Insufficienza renale severa (si 1; no 0)",
            'classe funzionale WHO-FC',
            'segni clinici di insufficienza cardiaca sinistra, astenia, dispnea (0, assenti; 1, presenti)',
            'Segni clinici di insufficienz acardiaca destra, edemi (0, assenti; 1, presenti)',
            'Qualsiasi segno di insufficienza cardiaca',
            'PAS (mmHg)', 'PAD (mmHg)', 'NT-pro-BNP (pg/mL)',
            'Fibrillazione atriale parossistica (0, assente; 1, presente)',
            'Fibrillazione atriale cronica (0, assente; 1, presente)', 'Fumo (0 no; 1 si)', 'sesso']



NUM_FEATURES = [
    'Epicardial fat thickness (mm)', 'LVEDD (mm)', 'LVESD (mm)', 'LV mass (g)',
    'LV mass i (g/m2) Calcolo automatico', 'LAD (mm)', 'LAV (ml)', 'iLAV (mL/m2) Calcolo automatico',
    'LVEF (%)', 'E/A', 'E/e avg', 'VCI diameter (mm)', 'sPAP (mmHg) = RV-RA + RAP (3-8-15)',
    'eta', 'BSA', 'PAS (mmHg)', 'PAD (mmHg)', 'NT-pro-BNP (pg/mL)'
]

CAT_FEATURES = [
    'MR 0=no, 1=min, 2=lieve, 3=mod, 4=sev',
    'Probabilità IP a riposo (1, bassa; 2, intermedia; 3, alta)',
    'IAS (0, assente; 1, presente)', 'Dislipidemia (0, assente; 1, presente, 2 s. metabolica)',
    'Diabete mellito (0, assente; 1, presente, 2 diabete complicato)',
    'Beta bloccante 1 si, 0 no', 'MRA 1 si, 0 no', 'ACE/ARB 1 si, 0 no',
    'Terapia antiscompenso assente (0), sub-ottimale (1-2), ottimale (3)',
    'Insufficienza renale lieve (si 1; no 0)', 'Insufficienza renale severa (si 1; no 0)',
    'classe funzionale WHO-FC',
    'segni clinici di insufficienza cardiaca sinistra, astenia, dispnea (0, assenti; 1, presenti)',
    'Segni clinici di insufficienz acardiaca destra, edemi (0, assenti; 1, presenti)',
    'Qualsiasi segno di insufficienza cardiaca',
    'Fibrillazione atriale parossistica (0, assente; 1, presente)',
    'Fibrillazione atriale cronica (0, assente; 1, presente)', 'Fumo (0 no; 1 si)',
    'sesso'
]


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
    if feature == "sesso":
        user_input[feature] = st.selectbox("Sesso:", options=["0", "1"])
    elif any(x in feature.lower() for x in ["assente", "presente", "no", "si", "0", "1", "2"]):
        user_input[feature] = st.number_input(f"{feature}:", step=1.0)
    else:
        user_input[feature] = st.number_input(f"{feature}:", step=0.1)

if st.button("🔍 Estimate 🔍"):
    try:
        input_df = pd.DataFrame([user_input])
        input_df["sesso"] = input_df["sesso"].astype('category')

        
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

        
        fig, ax = plt.subplots()
        sns.barplot(x=["Logistic", "Random Forest", "XGBoost"], y=[prob_log, prob_rf, prob_gb], palette="Set2", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Predicted Probability")
        ax.set_title("Model Comparison")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ {e}")
        st.success("💃🥳YOHOOOOOOOOOO, Low Risk of HFpEF 🥳💃")

    # Bar chart for comparison
    fig, ax = plt.subplots(figsize=(6, 5))
    models = ["Logistic Regression", "Random Forest", "XG Boosting"]
    probabilities = [prob_log, prob_rf, prob_gb]
    sns.barplot(x=models, y=probabilities, palette='mako', ax=ax)
    ax.set_title("Model Probability Comparison  ")
    ax.set_ylabel("HFpEF Probability ")
    st.pyplot(fig)
