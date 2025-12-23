import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Zomato Rating Predictor",
    page_icon="🍽️",
    layout="centered"
)

st.title("🍽️ Zomato Restaurant Rating Predictor")
st.caption("Machine Learning App | Random Forest Regressor")

# -----------------------------
# Load trained model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "zomato_rating_model.joblib"

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    else:
        st.warning("Training model for the first time. Please wait...")
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        import pandas as pd

        df = pd.read_csv(BASE_DIR / "data" / "zomato.csv")

        df = df[
            [
                "online_order",
                "book_table",
                "location",
                "rest_type",
                "cuisines",
                "approx_cost(for two people)",
                "votes",
                "rate"
            ]
        ]

        df["rate"] = df["rate"].replace(["NEW", "-"], None)
        df["rate"] = df["rate"].astype(str).str.split("/").str[0].astype(float)
        df.dropna(subset=["rate"], inplace=True)

        df["online_order"] = df["online_order"].map({"Yes": 1, "No": 0})
        df["book_table"] = df["book_table"].map({"Yes": 1, "No": 0})

        X = df.drop("rate", axis=1)
        y = df["rate"]

        cat_cols = X.select_dtypes(include="object").columns
        num_cols = X.select_dtypes(exclude="object").columns

        preprocessor = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols),
            ]
        )

        model = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )

        model.fit(X, y)
        return model

model = load_model()

# -----------------------------
# User Input Form
# -----------------------------
st.subheader("📌 Enter Restaurant Details")

online_order = st.selectbox(
    "Online Order Available?",
    options=["Yes", "No"]
)

book_table = st.selectbox(
    "Table Booking Available?",
    options=["Yes", "No"]
)

location = st.text_input(
    "Location",
    value="Banashankari"
)

rest_type = st.text_input(
    "Restaurant Type",
    value="Casual Dining"
)

cuisines = st.text_input(
    "Cuisines",
    value="North Indian"
)

approx_cost = st.number_input(
    "Approx Cost for Two People",
    min_value=50,
    max_value=10000,
    value=500
)

votes = st.number_input(
    "Number of Votes",
    min_value=0,
    max_value=100000,
    value=100
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔮 Predict Rating"):
    
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

    st.success(f"⭐ Predicted Restaurant Rating: **{prediction:.1f} / 5**")
