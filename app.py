import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load pre-trained model
@st.cache_resource
def load_model():
    return joblib.load("models/risk_model.pkl")

# Preprocess uploaded data
def preprocess(data):
    data = data.dropna()
    features = ['heart_rate', 'steps', 'sleep_hours']
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    return data, features

# Predict risk
def predict(data, model, features):
    data['risk_level'] = model.predict(data[features])
    return data

def main():
    st.title("Health Risk Prediction from Wearable Data")
    st.markdown("Upload your wearable data CSV file (e.g., from Fitbit)")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
        st.subheader("Raw Data")
        st.dataframe(df.head())

        df, features = preprocess(df)

        model = load_model()
        df = predict(df, model, features)

        st.subheader("Predicted Risk Levels")
        st.dataframe(df[['timestamp', 'heart_rate', 'steps', 'sleep_hours', 'risk_level']].head())

        # Visualization
        st.subheader("Risk Level Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='risk_level', ax=ax)
        st.pyplot(fig)

        st.subheader("Download Predictions")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Risk Predictions as CSV",
            data=csv,
            file_name='risk_predictions.csv',
            mime='text/csv'
        )

if __name__ == '__main__':
    main()