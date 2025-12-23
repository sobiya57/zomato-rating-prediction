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
st.caption(
    "An end-to-end Machine Learning application that predicts restaurant ratings "
    "based on customer behavior, pricing, and restaurant features."
)

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

# -----------------------------
# Helper Functions
# -----------------------------
def interpret_rating(rating: float):
    """
    Convert numeric rating into human-friendly explanation
    """
    if rating >= 4.5:
        return (
            "🌟 Excellent Restaurant",
            "Outstanding ratings indicate exceptional food quality, service, and customer satisfaction."
        )
    elif rating >= 4.0:
        return (
            "😄 Very Good Restaurant",
            "Customers generally love this restaurant. Strong votes, good pricing, and popular cuisine."
        )
    elif rating >= 3.5:
        return (
            "🙂 Good Restaurant",
            "This restaurant performs well overall with decent popularity and customer engagement."
        )
    elif rating >= 3.0:
        return (
            "😐 Average Restaurant",
            "The restaurant has mixed reviews. Improvements in service or pricing could help."
        )
    else:
        return (
            "⚠️ Below Average Restaurant",
            "Lower ratings suggest customer dissatisfaction or limited popularity."
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
        if not hasattr(model, "named_steps"):
            st.warning("⚠️ Feature importance not supported for this model.")
            return

        regressor = model.named_steps.get("regressor", None)
        preprocessor = model.named_steps.get("preprocessor", None)

        if regressor is None or not hasattr(regressor, "feature_importances_"):
            st.warning("⚠️ Feature importance not available.")
            return

        importances = regressor.feature_importances_

        # Get feature names after preprocessing
        feature_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out(cols))
            else:
                feature_names.extend(cols)

        import pandas as pd
        import matplotlib.pyplot as plt

        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(15)

        fig, ax = plt.subplots()
        ax.barh(fi_df["Feature"], fi_df["Importance"])
        ax.invert_yaxis()
        ax.set_title("Top Feature Importance")

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

# -------------------------------
# Rating Interpretation Function
# -------------------------------
def interpret_rating(rating):
    if rating >= 4.5:
        return "🌟 Excellent", "Highly rated restaurant with outstanding customer satisfaction."
    elif rating >= 4.0:
        return "😍 Very Good", "Customers generally love this restaurant."
    elif rating >= 3.5:
        return "👍 Good", "A reliable choice with positive reviews."
    elif rating >= 3.0:
        return "😐 Average", "Decent experience but has room for improvement."
    else:
        return "⚠️ Below Average", "Lower ratings – quality or service may be inconsistent."

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

# 👇 RIGHT PLACE (you added interpretation here)
label, explanation = interpret_rating(rating)
st.markdown("## 🧠 Rating Interpretation")
st.info(explanation)

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption(
    "📌 This is a real-world Machine Learning project built using Zomato restaurant data. "
    "The model is trained using RandomForest Regression and deployed on Streamlit Cloud."
)


