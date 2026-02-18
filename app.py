import streamlit as st
import pandas as pd
import joblib

# Load Model and Features
model = joblib.load("lasso_car_price_model.pkl")
features = joblib.load("model_features.pkl")

# Page Config
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# ---------- Custom CSS ----------
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #1f4e79;
    }
    .sub-text {
        text-align: center;
        color: gray;
        margin-bottom: 25px;
    }
    .prediction-box {
        background-color: #e8f4ff;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #0b5394;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">🚗 Car Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Enter car details to estimate selling price</div>', unsafe_allow_html=True)

# ---------- Input Section ----------
st.subheader("Car Details")

col1, col2 = st.columns(2)

with col1:
    present_price = st.number_input("💰 Present Price (Lakhs)", min_value=0.0)
    kms_driven = st.number_input("📍 Kilometers Driven", min_value=0)
    owner = st.selectbox("👤 Owner Type", [0, 1, 2, 3])

with col2:
    car_age = st.number_input("📅 Car Age (Years)", min_value=0)
    fuel_type = st.selectbox("⛽ Fuel Type", ["Petrol", "Diesel", "CNG"])
    transmission = st.selectbox("⚙️ Transmission", ["Manual", "Automatic"])

seller_type = st.selectbox("🏷️ Seller Type", ["Dealer", "Individual"])

st.markdown("---")

# Convert Inputs
input_dict = {
    "Present_price": present_price,
    "Kms_driven": kms_driven,
    "Owner": owner,
    "Car_age": car_age,
    "Fuel_type_Diesel": 1 if fuel_type == "Diesel" else 0,
    "Fuel_type_Petrol": 1 if fuel_type == "Petrol" else 0,
    "Fuel_type_CNG": 1 if fuel_type == "CNG" else 0,
    "Seller_type_Individual": 1 if seller_type == "Individual" else 0,
    "Transmission_Manual": 1 if transmission == "Manual" else 0,
    "Transmission_Automatic": 1 if transmission == "Automatic" else 0,
}

input_df = pd.DataFrame([input_dict])

# Align Features
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[features]

# ---------- Prediction Button ----------
st.markdown("###")

if st.button("🔮 Predict Price", use_container_width=True):
    prediction = model.predict(input_df)[0]

    st.markdown(
        f'<div class="prediction-box">Estimated Price: ₹ {round(prediction,2)} Lakhs</div>',
        unsafe_allow_html=True
    )
