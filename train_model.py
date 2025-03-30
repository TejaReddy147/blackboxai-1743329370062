import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Define the same mappings as in app.py
SYMPTOM_MAP = {
    'fever': 0, 'fatigue': 1, 'weight_loss': 2, 'headache': 3,
    'chest_pain': 4, 'abdominal_pain': 5, 'cough': 6,
    'shortness_breath': 7, 'sore_throat': 8, 'nausea': 9,
    'diarrhea': 10, 'vomiting': 11, 'dizziness': 12,
    'confusion': 13, 'seizures': 14, 'rash': 15,
    'itching': 16, 'swelling': 17
}

DISEASE_MAP = {
    0: 'Influenza (Flu)',
    1: 'Common Cold',
    2: 'COVID-19',
    3: 'Pneumonia',
    4: 'Bronchitis',
    5: 'Strep Throat',
    6: 'Allergic Reaction',
    7: 'Migraine',
    8: 'Gastroenteritis',
    9: 'Urinary Tract Infection'
}

# Create and train a simple model
model = RandomForestClassifier(n_estimators=10, random_state=42)
X = np.random.rand(100, len(SYMPTOM_MAP)+3)  # +3 for age and gender features
y = np.random.randint(0, len(DISEASE_MAP), 100)
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully")