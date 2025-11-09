import pandas as pd
import numpy as np

# Generate dataset for disease prediction
np.random.seed(42)
n_samples = 5000

data = {
    'age': np.random.randint(18, 85, n_samples),
    'gender': np.random.choice(['male', 'female'], n_samples),
    'ethnicity': np.random.choice(['caucasian', 'asian', 'african', 'hispanic', 'other'], n_samples),
    'occupation': np.random.choice(['office', 'manual', 'healthcare', 'education', 'retired'], n_samples),
    'diabetes': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    'hypertension': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
    'asthma': np.random.choice([0, 1], n_samples, p=[0.90, 0.10]),
    'heart_disease': np.random.choice([0, 1], n_samples, p=[0.88, 0.12]),
    'cancer': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    'fever': np.random.choice([0, 1], n_samples, p=[0.80, 0.20]),
    'cough': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
    'fatigue': np.random.choice([0, 1], n_samples, p=[0.70, 0.30]),
    'headache': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
    'chest_pain': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    'shortness_of_breath': np.random.choice([0, 1], n_samples, p=[0.82, 0.18]),
    'dizziness': np.random.choice([0, 1], n_samples, p=[0.78, 0.22]),
    'nausea': np.random.choice([0, 1], n_samples, p=[0.80, 0.20]),
    'blood_pressure_systolic': np.random.randint(90, 180, n_samples),
    'blood_pressure_diastolic': np.random.randint(60, 120, n_samples),
    'heart_rate': np.random.randint(50, 120, n_samples),
    'temperature': np.random.uniform(97.0, 103.0, n_samples),
    'respiratory_rate': np.random.randint(12, 25, n_samples),
    'oxygen_saturation': np.random.randint(88, 100, n_samples),
    'blood_glucose': np.random.randint(70, 250, n_samples),
    'cholesterol': np.random.randint(150, 350, n_samples),
    'hemoglobin': np.random.uniform(10.0, 18.0, n_samples),
    'white_blood_cells': np.random.randint(3000, 15000, n_samples),
    'physical_activity': np.random.choice(['low', 'moderate', 'high'], n_samples),
    'smoking': np.random.choice(['yes', 'no'], n_samples, p=[0.25, 0.75]),
    'alcohol': np.random.choice(['yes', 'no'], n_samples, p=[0.35, 0.65]),
    'sleep_hours': np.random.randint(4, 10, n_samples),
}

df = pd.DataFrame(data)

def assign_disease(row):
    if row['blood_glucose'] > 125 and ((row['diabetes'] == 1 or row['age'] > 45) or (row['fatigue']==1 and row['dizziness']==1)):
        return 'Diabetes'
    elif (row['chest_pain'] == 1 and (row['cholesterol'] > 240 and row['smoking'] == 'yes') or (row['heart_rate']<60 and row['dizziness']==1 and row['fatigue']==1) or (row['shortness_of_breath']==1 and row['chest_pain']==1)):
        return 'Heart Disease'
    elif (row['blood_pressure_systolic'] > 140 or row['blood_pressure_diastolic'] >90) or (row['hypertension'] == 1 and row['dizziness']==1 and row['headache']==1):
        return 'Hypertension'
    elif (row['fatigue']==1 and row['fever'] == 1 and row['shortness_of_breath']==1 or row['cough'] == 1 and row['fatigue'] == 1) or (row['asthma']==1 and row['oxygen_saturation']<95):
        return 'Respiratory Infection'
    else:
        return 'Healthy'

df['disease'] = df.apply(assign_disease, axis=1)
df.to_csv('disease_dataset.csv', index=False)
print(f"Dataset created with {len(df)} samples")
print(df['disease'].value_counts())