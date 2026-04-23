import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Title
st.title("📊 Bank Profitability Predictor (ROA)")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Indian_Banks.csv", encoding="latin1", engine="python")
    
    # 🔥 CLEAN COLUMN NAMES (THIS FIXES YOUR ERROR)
    df.columns = df.columns.str.replace("\n", " ")
    df.columns = df.columns.str.strip()
    
    return df

df = load_data()

# Show cleaned column names
st.subheader("Cleaned Columns")
st.write(df.columns.tolist())

# ✅ USE YOUR ACTUAL COLUMN NAMES
X = df[[
    "Net NPA Ratio (%)",
    "CAR / CRAR (%)"
]]

y = df["ROA (%) [DV]"]

# Train model
model = LinearRegression()
model.fit(X, y)

# User inputs
st.subheader("Enter Bank Details")

net_npa = st.number_input("Net NPA Ratio (%)", value=2.0)
car = st.number_input("CAR / CRAR (%)", value=12.0)

# Prediction
if st.button("Predict ROA"):
    input_data = np.array([[net_npa, car]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted ROA: {round(prediction, 3)} %")

# Chart
st.subheader("ROA Distribution")
st.bar_chart(y)