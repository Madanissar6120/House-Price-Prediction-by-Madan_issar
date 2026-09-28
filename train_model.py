"""
train_model.py
---------------
Full training pipeline for the House Price Predictor.

Workflow (this is exactly what to walk through in your review):
  1. Load data
  2. Handle missing values
  3. Feature engineering
  4. Train/test split
  5. Scale features
  6. Train two models (Linear Regression + Random Forest)
  7. Evaluate both, pick the better one
  8. Save the winning model + scaler + column order to disk
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
df = pd.read_csv("data/housing.csv")
print("Data shape:", df.shape)
print(df.isnull().sum())

# ------------------------------------------------------------------
# 2. HANDLE MISSING VALUES
# ------------------------------------------------------------------
# We use median imputation instead of mean because median is robust
# to outliers (a few very large houses won't skew the fill value).
df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())
df["garage"] = df["garage"].fillna(df["garage"].median())

# ------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ------------------------------------------------------------------
# total_rooms: a simple combined feature that often correlates well
# with price and can help linear models capture size in one number.
df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

# price_per_sqft is NOT created here as an input feature on purpose:
# it would leak the target (price) into the features, which is a
# classic beginner mistake ("data leakage") — be ready to explain this
# if asked "why didn't you engineer a price_per_sqft feature?".

FEATURE_COLUMNS = [
    "area_sqft", "bedrooms", "bathrooms", "stories",
    "age_years", "garage", "distance_to_city_km",
    "location_score", "total_rooms",
]
TARGET_COLUMN = "price"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# ------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ------------------------------------------------------------------
# 80/20 split, fixed random_state for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# 5. FEATURE SCALING
# ------------------------------------------------------------------
# Linear Regression is sensitive to feature scale (area_sqft is in the
# thousands, location_score is 1-10) so we standardize.
# Random Forest does NOT need scaling (it splits on raw thresholds),
# but we scale consistently for both to keep the pipeline simple and
# to make the saved scaler reusable for the Streamlit app.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 6. TRAIN MODELS
# ------------------------------------------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train_scaled, y_train)  # scaling is harmless for RF, kept for a single shared pipeline

# ------------------------------------------------------------------
# 7. EVALUATE
# ------------------------------------------------------------------
def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"\n{name}")
    print(f"  RMSE : {rmse:,.0f}")
    print(f"  MAE  : {mae:,.0f}")
    print(f"  R2   : {r2:.4f}")
    return rmse, mae, r2

lr_rmse, lr_mae, lr_r2 = evaluate("Linear Regression", lr_model, X_test_scaled, y_test)
rf_rmse, rf_mae, rf_r2 = evaluate("Random Forest", rf_model, X_test_scaled, y_test)

# ------------------------------------------------------------------
# 8. PICK BEST MODEL (lower RMSE wins) AND SAVE
# ------------------------------------------------------------------
if rf_rmse < lr_rmse:
    best_model, best_name = rf_model, "Random Forest"
else:
    best_model, best_name = lr_model, "Linear Regression"

print(f"\nSelected best model: {best_name}")

joblib.dump(best_model, "model/house_price_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(FEATURE_COLUMNS, "model/feature_columns.pkl")
joblib.dump(best_name, "model/model_name.pkl")

print("Saved model, scaler, and feature columns to /model")
