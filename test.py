import pandas as pd
import joblib

# Load saved model and scaler
model = joblib.load("Models/best_model.pkl")
scaler = joblib.load("Models/scaler.pkl")

# Feature names (must match training)
columns = [
    'Age', 'Sex', 'Constrictive pericarditis', 'Resting Blood Pressure',
    'Cholestrol', 'Fasting blood sugar', 'Resting electrocardiographic',
    'Maximum heart rate', 'Exercise induced angina', 'oldpeak',
    'slope', 'calcium', 'thalassemia'
]

# Fake patient names
patient_names = [
    "John Smith", "Emma Johnson", "Michael Brown", "Sophia Davis", "Liam Wilson",
    "Olivia Miller", "Noah Moore", "Ava Taylor", "Ethan Anderson", "Isabella Thomas"
]

# Sample data for each patient
sample_data = [
    [63, 1, 3, 150, 268, 1, 1, 187, 0, 3.6, 0, 2, 2],
    [45, 0, 2, 130, 220, 0, 1, 170, 0, 1.2, 1, 0, 2],
    [58, 1, 4, 140, 260, 1, 2, 150, 1, 2.3, 2, 2, 3],
    [39, 0, 0, 120, 204, 0, 0, 170, 0, 0.0, 1, 0, 2],
    [67, 1, 4, 160, 290, 1, 2, 120, 1, 3.2, 2, 2, 3],
    [52, 0, 1, 125, 230, 0, 1, 172, 0, 1.0, 1, 0, 2],
    [49, 1, 2, 138, 250, 1, 0, 155, 0, 1.4, 2, 0, 3],
    [60, 0, 3, 145, 275, 0, 1, 165, 0, 0.8, 1, 1, 2],
    [42, 0, 1, 125, 210, 0, 0, 190, 0, 0.1, 1, 0, 2],
    [55, 1, 3, 142, 260, 0, 1, 145, 1, 2.5, 2, 2, 3],
]

# Create DataFrame
df = pd.DataFrame(sample_data, columns=columns)

# Scale and predict
scaled_data = scaler.transform(df)
predictions = model.predict(scaled_data)

# Print results in a nice readable format
print("🩺 Heart Disease Prediction Results\n" + "-"*50)
for name, features, pred in zip(patient_names, sample_data, predictions):
    result = "💔 Has Heart Disease" if pred == 1 else "💚 No Heart Disease"
    print(f"{name:20s} → {result}")
