import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load sample data
df = pd.read_csv("sample_data.csv")

# Preprocessing
df = df.dropna()
features = ['heart_rate', 'steps', 'sleep_hours']
X = df[features]
y = df['risk_level']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, 'models/risk_model.pkl')
print("Model saved successfully to models/risk_model.pkl")
