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
    "BSA (non toccare)", 'Epicardial fat thickness (mm)', 'LVEDD (mm)', 'LVESD (mm)', 'LV mass (g)', 'LV mass i (g/m2) Calcolo automatico',
    'LAD (mm)', 'LAV (ml)', 'iLAV (mL/m2) Calcolo automatico', 'LVEF (%)', 'MR 0=no, 1=min, 2=lieve, 3=mod, 4=sev', 'E/A', 'E/e avg',
    'VCI diameter (mm)', 'sPAP (mmHg) = RV-RA + RAP (3-8-15)', 'Probabilità IP a riposo (1, bassa; 2, intermedia; 3, alta)', 'eta',
    'BSA', 'IAS (0, assente; 1, presente)', 'Dislipidemia (0, assente; 1, presente, 2 s. metabolica)', 'Diabete mellito (0, assente; 1, presente, 2 diabete complicato)',
    'Beta bloccante 1 si, 0 no', 'MRA 1 si, 0 no', 'ACE/ARB 1 si, 0 no', 'Terapia antiscompenso assente (0), sub-ottimale (1-2), ottimale (3)',
    "Insufficienza renale lieve (si 1; no 0)", "Insufficienza renale severa (si 1; no 0)", 'classe funzionale WHO-FC',
    'segni clinici di insufficienza cardiaca sinistra, astenia, dispnea (0, assenti; 1, presenti)',
    'Segni clinici di insufficienz acardiaca destra, edemi (0, assenti; 1, presenti)',
    'Qualsiasi segno di insufficienza cardiaca', 'PAS (mmHg)', 'PAD (mmHg)', 'NT-pro-BNP (pg/mL)',
    'Fibrillazione atriale parossistica (0, assente; 1, presente)', 'Fibrillazione atriale cronica (0, assente; 1, presente)',
    'Fumo (0 no; 1 si)', 'sesso'
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
