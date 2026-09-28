"""
app.py
------
Streamlit front-end for the House Price Predictor.

Run locally with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ------------------------------------------------------------------
# Load the trained model, scaler, and expected feature order.
# These files are produced by train_model.py — app.py never trains
# anything itself, it only loads what was already trained.
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Page setup — MUST be the very first Streamlit command in the script,
# before any other st.* call (including cached functions, which show
# a loading spinner that itself counts as a Streamlit command).
# ------------------------------------------------------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

@st.cache_resource
def load_artifacts():
    model_dir = Path(__file__).resolve().parent / "model"
    model = joblib.load(model_dir / "house_price_model.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")
    feature_columns = joblib.load(model_dir / "feature_columns.pkl")
    model_name = joblib.load(model_dir / "model_name.pkl")
    return model, scaler, feature_columns, model_name

model, scaler, feature_columns, model_name = load_artifacts()
st.title("🏠 House Price Predictor")
st.caption(f"Model in use: **{model_name}**")
st.write(
    "Enter the details of a house below and get an estimated market price. "
    "This model was trained on a simulated housing dataset."
)

st.divider()

# ------------------------------------------------------------------
# Input widgets — one for each feature the model expects
# ------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    area_sqft = st.number_input("Area (sq. ft.)", min_value=300, max_value=10000, value=1800, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
    stories = st.number_input("Stories (floors)", min_value=1, max_value=15, value=2, step=1)

with col2:
    age_years = st.number_input("Age of house (years)", min_value=0, max_value=100, value=10, step=1)
    garage = st.number_input("Garage spaces", min_value=0, max_value=5, value=1, step=1)
    distance_to_city_km = st.number_input("Distance to city center (km)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
    location_score = st.slider("Location score (1 = poor, 10 = prime)", min_value=1, max_value=10, value=6)

st.divider()

# ------------------------------------------------------------------
# Predict button
# ------------------------------------------------------------------
if st.button("Predict Price", type="primary", use_container_width=True):
    # Feature engineering must exactly match train_model.py
    total_rooms = bedrooms + bathrooms

    input_dict = {
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "age_years": age_years,
        "garage": garage,
        "distance_to_city_km": distance_to_city_km,
        "location_score": location_score,
        "total_rooms": total_rooms,
    }

    # Build the input row in the SAME column order the model was trained on.
    input_df = pd.DataFrame([input_dict])[feature_columns]

    # Apply the same scaler that was fit during training.
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    st.success(f"### Estimated Price: ₹ {prediction:,.0f}")
    st.caption("This is an estimate based on a trained regression model, not a guaranteed valuation.")

    with st.expander("See the exact inputs sent to the model"):
        st.dataframe(input_df)

# ------------------------------------------------------------------
# Sidebar: quick project info for reviewers
# ------------------------------------------------------------------
with st.sidebar:
    st.header("About this project")
    st.write(
        "- **Dataset:** Simulated housing data (2000 rows), generated with "
        "known feature-price relationships for reproducibility.\n"
        "- **Preprocessing:** Median imputation for missing values, "
        "StandardScaler for feature scaling.\n"
        "- **Feature engineering:** `total_rooms` = bedrooms + bathrooms.\n"
        "- **Models compared:** Linear Regression vs Random Forest "
        "(selected by lowest RMSE on a held-out test set).\n"
        "- **Metrics:** RMSE, MAE, R²."
    )
