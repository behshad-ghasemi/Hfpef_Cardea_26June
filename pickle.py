from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load models and pipeline
pipeline = joblib.load('data_with_pca.joblib')
logistic_model = joblib.load('Logistic_model.joblib')
random_forest_model = joblib.load('randomforest_model.joblib')
xgboost_model = joblib.load('xgboost_model.joblib')

# Define feature list
FEATURES = [ 
    'Epicardial fat thickness (mm)', 'LV mass i (g/m2) Calcolo automatico',
    'LAD (mm)','LVEF (%)','E/e avg', 'DM','WHO-FC','PAS (mmHg)', 'NT-pro-BNP (pg/mL)', 'AF'
]

@app.route('/')
def home():
    return render_template('index.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {feature: float(request.form[feature]) for feature in FEATURES}
    user_input_df = pd.DataFrame([input_data])
    user_input_df['sesso'] = user_input_df['sesso'].astype('category')

    # Transform input
    X_user_transformed = pipeline.transform(user_input_df)

    # Predictions
    threshold = 0.15
    predictions = {}

    def predict_model(name, model):
        try:
            prob = model.predict_proba(X_user_transformed)[:, 1][0]
            result = "HfpEf" if prob >= threshold else "Not HfpEf"
        except AttributeError:
            pred = model.predict(X_user_transformed)[0]
            result = "HfpEf" if pred == 1 else "Not HfpEf"
            prob = None
        return result, prob

    predictions['Logistic'], prob_log = predict_model('Logistic', logistic_model)
    predictions['Random Forest'], prob_rf = predict_model('Random Forest', random_forest_model)
    predictions['XGBoost'], prob_xgb = predict_model('XGBoost', xgboost_model)

    return render_template('index.html',
                           predictions=predictions,
                           prob_log=prob_log,
                           prob_rf=prob_rf,
                           prob_xgb=prob_xgb,
                           features=FEATURES)

if __name__ == '__main__':
    app.run(debug=True)
