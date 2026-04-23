import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Title
st.title("📊 Bank Profitability Predictor (ROA)")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Indian_Banks.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())

# Selecting variables (change names if needed based on your dataset)
X = df[["Gross NPA (%)", "Net NPA (%)", "Capital Adequacy Ratio (%)"]]
y = df["ROA (%)"]

# Train model
model = LinearRegression()
model.fit(X, y)

st.subheader("Enter Bank Details")

gross_npa = st.number_input("Gross NPA (%)", value=5.0)
net_npa = st.number_input("Net NPA (%)", value=2.0)
car = st.number_input("Capital Adequacy Ratio (%)", value=12.0)

# Prediction
if st.button("Predict ROA"):
    input_data = np.array([[gross_npa, net_npa, car]])
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted ROA: {round(prediction, 3)} %")

# Simple visualization (NO matplotlib used)
st.subheader("ROA Distribution")
st.bar_chart(df["ROA (%)"])