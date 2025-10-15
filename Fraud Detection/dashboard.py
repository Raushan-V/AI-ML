import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("Credit Card Fraud Detection Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('../creditcard.csv')
    return df

df = load_data()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "EDA", "Model Prediction"])

if page == "Overview":
    st.header("Dataset Overview")
    st.write(f"Total transactions: {len(df)}")
    st.write(f"Fraud transactions: {df['Class'].sum()}")
    st.write(f"Normal transactions: {len(df) - df['Class'].sum()}")
    st.write(f"Fraud percentage: {df['Class'].mean() * 100:.2f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class Distribution")
        fig, ax = plt.subplots()
        df['Class'].value_counts().plot(kind='bar', ax=ax)
        ax.set_xticklabels(['Normal', 'Fraud'])
        st.pyplot(fig)

    with col2:
        st.subheader("Amount Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['Amount'], bins=50, ax=ax)
        st.pyplot(fig)

elif page == "EDA":
    st.header("Exploratory Data Analysis")

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Time series
    st.subheader("Transactions Over Time")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(df['Time'], df['Amount'], c=df['Class'], alpha=0.5, s=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amount')
    st.pyplot(fig)

elif page == "Model Prediction":
    st.header("Fraud Prediction")

    st.write("Enter transaction details:")

    col1, col2, col3 = st.columns(3)

    with col1:
        time = st.number_input("Time", value=0.0)
        v1 = st.number_input("V1", value=0.0)
        v2 = st.number_input("V2", value=0.0)
        v3 = st.number_input("V3", value=0.0)
        v4 = st.number_input("V4", value=0.0)
        v5 = st.number_input("V5", value=0.0)
        v6 = st.number_input("V6", value=0.0)
        v7 = st.number_input("V7", value=0.0)
        v8 = st.number_input("V8", value=0.0)
        v9 = st.number_input("V9", value=0.0)
        v10 = st.number_input("V10", value=0.0)

    with col2:
        v11 = st.number_input("V11", value=0.0)
        v12 = st.number_input("V12", value=0.0)
        v13 = st.number_input("V13", value=0.0)
        v14 = st.number_input("V14", value=0.0)
        v15 = st.number_input("V15", value=0.0)
        v16 = st.number_input("V16", value=0.0)
        v17 = st.number_input("V17", value=0.0)
        v18 = st.number_input("V18", value=0.0)
        v19 = st.number_input("V19", value=0.0)
        v20 = st.number_input("V20", value=0.0)

    with col3:
        v21 = st.number_input("V21", value=0.0)
        v22 = st.number_input("V22", value=0.0)
        v23 = st.number_input("V23", value=0.0)
        v24 = st.number_input("V24", value=0.0)
        v25 = st.number_input("V25", value=0.0)
        v26 = st.number_input("V26", value=0.0)
        v27 = st.number_input("V27", value=0.0)
        v28 = st.number_input("V28", value=0.0)
        amount = st.number_input("Amount", value=0.0)

    if st.button("Predict"):
        transaction_data = {
            "Time": time,
            "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5, "V6": v6, "V7": v7, "V8": v8, "V9": v9, "V10": v10,
            "V11": v11, "V12": v12, "V13": v13, "V14": v14, "V15": v15, "V16": v16, "V17": v17, "V18": v18, "V19": v19, "V20": v20,
            "V21": v21, "V22": v22, "V23": v23, "V24": v24, "V25": v25, "V26": v26, "V27": v27, "V28": v28,
            "Amount": amount
        }

        # For demo purposes, simulate prediction
        prediction = 0  # Normal
        probability = 0.02

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("🚨 Fraud Detected!")
        else:
            st.success("✅ Transaction appears normal")

        st.write(f"Fraud Probability: {probability:.4f}")

        # Connect to API for prediction
        try:
            response = requests.post("http://localhost:8000/predict", json=transaction_data)
            result = response.json()
            st.subheader("Prediction Result")
            if result["prediction"] == 1:
                st.error("🚨 Fraud Detected!")
            else:
                st.success("✅ Transaction appears normal")
            st.write(f"Fraud Probability: {result['fraud_probability']:.4f}")
            st.write(f"Message: {result['message']}")
        except requests.exceptions.RequestException:
            st.error("API not running. Start the API server first with: `uvicorn app:app --reload`")

        # Monitoring section
        st.subheader("Model Monitoring")
        try:
            monitor_response = requests.get("http://localhost:8000/monitoring/report")
            if monitor_response.status_code == 200:
                report = monitor_response.json()["report"]
                st.text_area("Monitoring Report", report, height=200)
            else:
                st.warning("Could not fetch monitoring report")
        except:
            st.warning("Monitoring service not available")
