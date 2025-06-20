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

# اصلاح 1: مطابقت نام‌های ویژگی‌ها
FEATURES = [
    'Epicardial fat thickness (mm)', 
    'LV mass i (g/m2) Calcolo automatico',
    'LAD (mm)',
    'LVEF (%)',
    'E/e avg', 
    'DM',
    'WHO-FC',
    'PAS (mmHg)', 
    'NT-pro-BNP (pg/mL)', 
    'AF'  # اصلاح: AF به جای FA
]

NUM_FEATURES = [
    'Epicardial fat thickness (mm)',
    'LV mass i (g/m2) Calcolo automatico', 
    'LAD (mm)',
    'LVEF (%)',
    'E/e avg', 
    'PAS (mmHg)', 
    'NT-pro-BNP (pg/mL)'
]

# اصلاح 2: AF به جای FA
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

# اصلاح 3: ورودی‌های مناسب برای ویژگی‌های کتگوریکال
user_input = {}

# ویژگی‌های عددی
for feature in NUM_FEATURES:
    if feature == 'LVEF (%)':
        user_input[feature] = st.number_input(f"{feature}:", min_value=0.0, max_value=100.0, step=0.1)
    elif feature == 'PAS (mmHg)':
        user_input[feature] = st.number_input(f"{feature}:", min_value=0.0, max_value=300.0, step=1.0)
    elif feature == 'NT-pro-BNP (pg/mL)':
        user_input[feature] = st.number_input(f"{feature}:", min_value=0.0, step=1.0)
    else:
        user_input[feature] = st.number_input(f"{feature}:", step=0.1)

# ویژگی‌های کتگوریکال
user_input['DM'] = st.selectbox("Diabetes Mellitus (DM):", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
user_input['WHO-FC'] = st.selectbox("WHO Functional Class:", options=[1, 2, 3, 4])
user_input['AF'] = st.selectbox("Atrial Fibrillation (AF):", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

if st.button("🔍 Estimate 🔍"):
    try:
        input_df = pd.DataFrame([user_input])
        
        # اصلاح 4: بررسی وجود ویژگی‌ها قبل از تبدیل
        missing_features = set(FEATURES) - set(input_df.columns)
        if missing_features:
            st.error(f"Missing features: {missing_features}")
            st.stop()
        
        # مرتب کردن ستون‌ها
        input_df = input_df[FEATURES]
        
        transformed_input = pipeline.transform(input_df)

        # محاسبه احتمالات
        prob_log = log_model.predict_proba(transformed_input)[0][1]
        prob_rf = rf_model.predict_proba(transformed_input)[0][1]
        prob_gb = xgb_model.predict_proba(transformed_input)[0][1]

        st.subheader("🤔 Prediction Probabilities")
        st.write(f"🔹 **Logistic Regression**: `{prob_log:.4f}`")
        st.write(f"🔹 **Random Forest**: `{prob_rf:.4f}`")
        st.write(f"🔹 **XG Boosting**: `{prob_gb:.4f}`")

        # اصلاح 5: تغییر آستانه تصمیم‌گیری
        if prob_gb > 0.5:  # معمولاً 0.5 بهتر از 0.6 است
            st.error("🚨💀 High Risk of HFpEF Detected! 😱")
        else:
            st.success("✅ Low Risk of HFpEF Detected 🎉")

        # نمودار مقایسه مدل‌ها
        fig, ax = plt.subplots(figsize=(10, 6))
        models = ["Logistic Regression", "Random Forest", "XG Boosting"]
        probabilities = [prob_log, prob_rf, prob_gb]
        bars = ax.bar(models, probabilities, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_title("Model Probability Comparison", fontsize=14, fontweight='bold')
        ax.set_ylabel("HFpEF Probability", fontsize=12)
        ax.set_ylim(0, 1)
        
        # اضافه کردن خط آستانه
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold (0.5)')
        ax.legend()
        
        # اضافه کردن مقادیر روی نمودار
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

        # اصلاح 6: SHAP analysis برای XGBoost
        try:
            st.subheader("🎯 SHAP Explanation for this Patient (XGBoost)")
            
            # تبدیل داده‌ها قبل از PCA برای SHAP
            features_before_pca = pipeline.named_steps['preprocessing'].get_feature_names_out()
            df_for_shap = pd.DataFrame(
                pipeline.named_steps['preprocessing'].transform(input_df), 
                columns=features_before_pca
            )
            
            # ایجاد explainer و محاسبه SHAP values
            explainer = shap.Explainer(xgb_model)
            shap_values = explainer(df_for_shap)
            
            # نمودار SHAP waterfall
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(fig)
            plt.close()
            
        except Exception as shap_error:
            st.warning(f"SHAP analysis could not be performed: {shap_error}")
            st.info("This might be due to model incompatibility with SHAP or preprocessing issues.")

        # اصلاح 7: Feature importance برای مدل‌های tree-based
        def plot_feature_importance(model, model_name):
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                # استفاده از نام‌های ویژگی پس از preprocessing
                feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
                
                if len(importance) == len(feature_names):
                    importance_df = pd.DataFrame({
                        "Feature": feature_names, 
                        "Importance": importance
                    })
                else:
                    # اگر PCA اعمال شده باشد
                    feature_names = [f"PC_{i+1}" for i in range(len(importance))]
                    importance_df = pd.DataFrame({
                        "Feature": feature_names, 
                        "Importance": importance
                    })
                
                importance_df = importance_df.sort_values(by="Importance", ascending=False)
                
                st.write(f"### Feature Importance - {model_name}")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(data=importance_df.head(10), x="Importance", y="Feature", ax=ax)
                ax.set_title(f"Top 10 Feature Importance - {model_name}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                return fig
            else:
                st.warning(f"{model_name} does not have feature_importances_ attribute")
                return None

        # نمایش feature importance برای مدل‌های مختلف
        plot_feature_importance(rf_model, "Random Forest")
        plot_feature_importance(xgb_model, "XGBoost")

    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}")
        st.error("Please check your input values and try again.")
        
    finally:
        # بستن تمام figures برای جلوگیری از memory leak
        plt.close('all')
