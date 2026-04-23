import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Title
st.title("📊 Bank Profitability Predictor (ROA)")

# Load dataset (FIX APPLIED HERE 👇)
@st.cache_data
def load_data():
    df = pd.read_csv("Indian_Banks.csv", encoding="latin1", engine="python")
    return df

df = load_data()

# Show dataset
st.subheader("Dataset Preview")
st.write(df.head())

# Select variables (MAKE SURE COLUMN NAMES MATCH YOUR CSV)
X = df[["Gross NPA (%)", "Net NPA (%)", "Capital Adequacy Ratio (%)"]]
y = df["ROA (%)"]

# Train model
model = LinearRegression()
model.fit(X, y)

# User inputs
st.subheader("Enter Bank Details")

gross_npa = st.number_input("Gross NPA (%)", value=5.0)
net_npa = st.number_input("Net NPA (%)", value=2.0)
car = st.number_input("Capital Adequacy Ratio (%)", value=12.0)

# Prediction
if st.button("Predict ROA"):
    input_data = np.array([[gross_npa, net_npa, car]])
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted ROA: {round(prediction, 3)} %")

# Simple chart (no matplotlib → no errors)
st.subheader("ROA Distribution")
st.bar_chart(df["ROA (%)"])