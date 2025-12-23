import streamlit as st
import pandas as pd
import joblib
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------
# App Config
# ----------------------------------
st.set_page_config(
    page_title="Zomato Rating Predictor",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Zomato Restaurant Rating Predictor")
st.caption("Predict restaurant ratings using machine learning")

# ----------------------------------
# Paths & Model URL
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "zomato_rating_model.joblib"

MODEL_URL = (
    "https://github.com/sobiya57/zomato-rating-prediction/"
    "releases/download/v1.0-model/zomato_rating_model.joblib"
)

# ----------------------------------
# Download Model if Needed
# ----------------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    if not MODEL_PATH.exists():
        with st.spinner("⬇️ Downloading ML model (first run only)..."):
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            MODEL_PATH.write_bytes(response.content)

    return joblib.load(MODEL_PATH)

# ----------------------------------
# Load Model
# ----------------------------------
model = load_model()
def show_feature_importance(model):
    try:
        preprocessor = model.named_steps["preprocessor"]
        regressor = model.named_steps["regressor"]

        # Get feature names
        num_features = preprocessor.transformers_[0][2]
        cat_features = preprocessor.transformers_[1][2]

        ohe = preprocessor.transformers_[1][1]
        cat_feature_names = ohe.get_feature_names_out(cat_features)

        feature_names = np.concatenate([num_features, cat_feature_names])

        importances = regressor.feature_importances_

        # Top 10 features
        indices = np.argsort(importances)[-10:]
        top_features = feature_names[indices]
        top_importances = importances[indices]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top_features, top_importances)
        ax.set_title("Top 10 Feature Importances")
        ax.set_xlabel("Importance Score")

        st.pyplot(fig)

    except Exception as e:
        st.warning("⚠️ Feature importance not available.")

# ----------------------------------
# Sidebar Inputs
# ----------------------------------
st.sidebar.header("📋 Restaurant Details")

online_order = st.sidebar.selectbox(
    "Online Order Available?", ["Yes", "No"]
)

book_table = st.sidebar.selectbox(
    "Table Booking Available?", ["Yes", "No"]
)

location = st.sidebar.text_input(
    "Location", value="Bangalore"
)

rest_type = st.sidebar.text_input(
    "Restaurant Type", value="Casual Dining"
)

cuisines = st.sidebar.text_input(
    "Cuisines", value="North Indian, Chinese"
)

approx_cost = st.sidebar.number_input(
    "Approx Cost for Two (₹)",
    min_value=50,
    max_value=10000,
    value=500,
    step=50
)

votes = st.sidebar.number_input(
    "Number of Votes",
    min_value=0,
    max_value=100000,
    value=100,
    step=10
)

# ----------------------------------
# Prediction Button
# ----------------------------------
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

    prediction = model.predict(input_df)[0]

    st.success(
        f"⭐ **Predicted Restaurant Rating: {prediction:.1f} / 5**"
    )
st.subheader("📊 Why this rating?")
show_feature_importance(model)

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption(
    "📌 This is a real-world machine learning project built using Zomato data "
    "and deployed on Streamlit Cloud."
)
