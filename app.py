from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import json

app = Flask(__name__)
CORS(app)

print("Loading model and preprocessors...")
try:
    model = joblib.load('disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    
    with open('model_info.json', 'r') as f:
        model_info = json.load(f)
    
    print("SUCCESS: Model loaded!")
    print(f"Model accuracy: {model_info['accuracy'] * 100:.2f}%")
    print(f"Target classes: {model_info['target_classes']}")
except Exception as e:
    print(f"ERROR: {e}")
    exit(1)

def safe_int(value, default=0):
    """Safely convert value to int, return default if fails"""
    try:
        if value is None or str(value).strip() == '':
            return default
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_float(value, default=0.0):
    """Safely convert value to float, return default if fails"""
    try:
        if value is None or str(value).strip() == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def preprocess_input(data):
    processed_data = {
        'age': safe_int(data.get('age'), 30),
        'gender': str(data.get('gender', 'male')).lower(),
        'ethnicity': str(data.get('ethnicity', 'caucasian')).lower(),
        'occupation': str(data.get('occupation', 'office')).lower(),
        'diabetes': int(data.get('pastIllnesses', {}).get('diabetes', False)),
        'hypertension': int(data.get('pastIllnesses', {}).get('hypertension', False)),
        'asthma': int(data.get('pastIllnesses', {}).get('asthma', False)),
        'heart_disease': int(data.get('pastIllnesses', {}).get('heartDisease', False)),
        'cancer': int(data.get('pastIllnesses', {}).get('cancer', False)),
        'fever': int(data.get('symptoms', {}).get('fever', False)),
        'cough': int(data.get('symptoms', {}).get('cough', False)),
        'fatigue': int(data.get('symptoms', {}).get('fatigue', False)),
        'headache': int(data.get('symptoms', {}).get('headache', False)),
        'chest_pain': int(data.get('symptoms', {}).get('chestPain', False)),
        'shortness_of_breath': int(data.get('symptoms', {}).get('shortnessOfBreath', False)),
        'dizziness': int(data.get('symptoms', {}).get('dizziness', False)),
        'nausea': int(data.get('symptoms', {}).get('nausea', False)),
        'blood_pressure_systolic': 120,
        'blood_pressure_diastolic': 80,
        'heart_rate': safe_int(data.get('heartRate'), 72),
        'temperature': safe_float(data.get('temperature'), 98.6),
        'respiratory_rate': safe_int(data.get('respiratoryRate'), 16),
        'oxygen_saturation': safe_int(data.get('oxygenSaturation'), 98),
        'blood_glucose': safe_int(data.get('bloodGlucose'), 100),
        'cholesterol': safe_int(data.get('cholesterol'), 200),
        'hemoglobin': safe_float(data.get('hemoglobin'), 14.0),
        'white_blood_cells': safe_int(data.get('whiteBloodCells'), 7000),
        'physical_activity': str(data.get('physicalActivity', 'moderate')).lower(),
        'smoking': str(data.get('smoking', 'no')).lower(),
        'alcohol': str(data.get('alcohol', 'no')).lower(),
        'sleep_hours': safe_int(data.get('sleepHours'), 7),
    }
    
    if 'bloodPressure' in data and data['bloodPressure']:
        try:
            bp = str(data['bloodPressure']).split('/')
            if len(bp) == 2:
                processed_data['blood_pressure_systolic'] = safe_int(bp[0], 120)
                processed_data['blood_pressure_diastolic'] = safe_int(bp[1], 80)
        except:
            pass
    
    return processed_data

def get_risk_factors(data):
    risk_factors = []
    if safe_int(data.get('bloodGlucose')) > 125:
        risk_factors.append('Elevated blood glucose levels')
    if safe_int(data.get('cholesterol')) > 240:
        risk_factors.append('High cholesterol')
    if safe_int(data.get('age')) > 50:
        risk_factors.append('Age factor (>50 years)')
    if data.get('smoking', 'no') == 'yes':
        risk_factors.append('Smoking habit')
    return risk_factors if risk_factors else ['Minimal risk factors detected']

def get_recommendations(prediction):
    recommendations = {
        'Diabetes': [
            'Consult an endocrinologist immediately',
            'Monitor blood glucose levels daily',
            'Adopt a low-carb, balanced diet',
            'Increase physical activity'
        ],
        'Heart Disease': [
            'Visit cardiologist for ECG',
            'Heart-healthy diet',
            'Quit smoking',
            'Monitor blood pressure'
        ],
        'Hypertension': [
            'Reduce sodium intake',
            'Exercise regularly',
            'Monitor blood pressure',
            'Reduce stress'
        ],
        'Respiratory Infection': [
            'Rest and isolate',
            'Stay hydrated',
            'Monitor oxygen levels',
            'Seek medical attention if needed'
        ],
        'Healthy': [
            'Maintain current lifestyle',
            'Annual checkups',
            'Regular exercise',
            'Balanced diet'
        ]
    }
    return recommendations.get(prediction, recommendations['Healthy'])

@app.route('/')
def home():
    return jsonify({'message': 'Disease Prediction API', 'status': 'running'})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        processed_data = preprocess_input(data)
        df = pd.DataFrame([processed_data])
        
        for col in model_info['categorical_cols']:
            if col in df.columns:
                le = label_encoders[col]
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])
        
        df[model_info['numerical_cols']] = scaler.transform(df[model_info['numerical_cols']])
        
        prediction_encoded = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0]
        disease = target_encoder.inverse_transform([prediction_encoded])[0]
        confidence = float(np.max(prediction_proba) * 100)
        
        risk_level = 'High' if confidence > 80 else 'Moderate' if confidence > 60 else 'Low'
        
        response = {
            'disease': disease,
            'confidence': round(confidence, 2),
            'riskLevel': risk_level,
            'riskFactors': get_risk_factors(data),
            'recommendations': get_recommendations(disease),
            'bmi': round(22 + np.random.random() * 6, 1)
        }
        
        print(f"✓ Prediction: {disease} ({confidence:.2f}%)")
        return jsonify(response)
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("" + "="*50)
    print("Flask server starting on http://localhost:5000")
    print("="*50 + "")
    app.run(debug=True, port=5000)