import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

print("Loading dataset...")
df = pd.read_csv('disease_dataset.csv')

X = df.drop('disease', axis=1)
y = df['disease']

print("Encoding categorical variables...")
label_encoders = {}
categorical_cols = ['gender', 'ethnicity', 'occupation', 'physical_activity', 'smoking', 'alcohol']

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print("Scaling features...")
scaler = StandardScaler()
numerical_cols = ['age', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
                  'heart_rate', 'temperature', 'respiratory_rate', 'oxygen_saturation',
                  'blood_glucose', 'cholesterol', 'hemoglobin', 'white_blood_cells', 'sleep_hours']

X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print("Training model...")
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("Saving model...")
joblib.dump(model, 'disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(le_target, 'target_encoder.pkl')

model_info = {
    'feature_names': list(X.columns),
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
    'target_classes': list(le_target.classes_),
    'accuracy': float(accuracy)
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)

print("Training complete! Files created.")