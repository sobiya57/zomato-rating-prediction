import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(
    page_title="Zomato Rating Predictor",
    layout="centered"
)

st.title("🍽️ Zomato Restaurant Rating Predictor")
st.caption("Predict restaurant ratings using machine learning")

# -----------------------------
# Base Directory
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "zomato_rating_model.joblib"

# -----------------------------
# Load Model (Deployment Safe)
# -----------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("📋 Restaurant Details")

online_order = st.sidebar.selectbox("Online Order Available?", ["Yes", "No"])
book_table = st.sidebar.selectbox("Table Booking Available?", ["Yes", "No"])

location = st.sidebar.text_input("Location", "Bangalore")
rest_type = st.sidebar.text_input("Restaurant Type", "Casual Dining")
cuisines = st.sidebar.text_input("Cuisines", "North Indian, Chinese")

approx_cost = st.sidebar.number_input(
    "Approx Cost for Two (₹)",
    min_value=50,
    max_value=5000,
    value=500
)

votes = st.sidebar.number_input(
    "Number of Votes",
    min_value=0,
    max_value=50000,
    value=100
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("⭐ Predict Rating"):

    input_df = pd.DataFrame([{
        "online_order": 1 if online_order == "Yes" else 0,
        "book_table": 1 if book_table == "Yes" else 0,
        "location": location,
        "rest_type": rest_type,
        "cuisines": cuisines,
        "approx_cost(for two people)": approx_cost,
        "votes": votes
    }])

    if model is None:
        st.warning("⚠️ Model file not found on server.")
        st.info("This app is running in demo mode. Prediction is disabled.")
    else:
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"⭐ Predicted Restaurant Rating: **{prediction:.1f} / 5**")
        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "📌 **Note:** This is a machine learning demo project built using Zomato data."
)
