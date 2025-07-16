import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

# Streamlit page config
st.set_page_config(page_title="Invoice Anomaly Detector", layout="wide")
st.title("📄 Invoice Anomaly Detector")
st.markdown("Upload your invoice data to detect suspicious or fraudulent transactions using machine learning.")

# File upload
uploaded_file = st.file_uploader("📤 Upload your invoice CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data")
    st.dataframe(df)

    # Anomaly Detection
    with st.spinner("🔍 Analyzing for anomalies..."):
        model = IsolationForest(contamination=0.1, random_state=42)
        df['anomaly_score'] = model.fit_predict(df[['amount']])
        df['Anomaly'] = df['anomaly_score'].apply(lambda x: '🚨 Yes' if x == -1 else '✅ No')

    st.subheader("📌 Detection Results")
    st.dataframe(df[['invoice_id', 'client_name', 'amount', 'date', 'Anomaly']])

    # Download results
    st.download_button(
        "📥 Download Anomaly Report",
        df.to_csv(index=False).encode('utf-8'),
        file_name="anomaly_report.csv",
        mime="text/csv"
    )
else:
    st.info("⬆️ Please upload a CSV file to begin.")
