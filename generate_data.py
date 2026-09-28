"""
generate_data.py
-----------------
Generates a synthetic but realistic housing dataset.

WHY SYNTHETIC DATA?
Public housing datasets (e.g. the old "Boston Housing" dataset) are either
deprecated for ethical reasons or require an internet download to fetch.
Here we generate data ourselves using a fixed random seed, so:
  1. The project runs 100% offline / reproducibly.
  2. We KNOW the true relationship between features and price
     (area increases price, age decreases it, etc.), which makes it
     easy to sanity-check that the model is learning something real.

Run this once to create data/housing.csv
"""

import numpy as np
import pandas as pd

# Fixed seed -> same dataset every time we run this script (reproducibility)
np.random.seed(42)

N = 2000  # number of houses to simulate

# ---- Feature generation ----
area_sqft = np.random.normal(1800, 600, N).clip(400, 6000)
bedrooms = np.random.randint(1, 6, N)
bathrooms = np.random.randint(1, 4, N)
stories = np.random.randint(1, 4, N)
age_years = np.random.randint(0, 50, N)
garage = np.random.randint(0, 3, N)              # number of garage spaces
distance_to_city_km = np.random.uniform(0.5, 40, N)
location_score = np.random.randint(1, 11, N)      # 1 (poor) - 10 (prime) locality rating

# ---- Target generation (price) ----
# We build price from a linear combination of features PLUS random noise,
# so the relationships are realistic but not perfectly clean (like real life).
price = (
    area_sqft * 120                     # bigger house -> more expensive
    + bedrooms * 8000
    + bathrooms * 6000
    + stories * 4000
    - age_years * 900                   # older house -> cheaper
    + garage * 5000
    - distance_to_city_km * 1500        # farther from city -> cheaper
    + location_score * 9000             # better locality -> more expensive
    + 50000                             # base price
)

# Add random noise (market randomness, negotiation, etc.)
noise = np.random.normal(0, 25000, N)
price = (price + noise).clip(30000, None)  # no negative/absurdly low prices

df = pd.DataFrame({
    "area_sqft": area_sqft.round(0).astype(int),
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "age_years": age_years,
    "garage": garage,
    "distance_to_city_km": distance_to_city_km.round(1),
    "location_score": location_score,
    "price": price.round(0).astype(int),
})

# ---- Simulate a little real-world messiness ----
# Real datasets almost always have a few missing values.
# We intentionally null out ~1% of a couple columns so the training
# script has genuine preprocessing to do (and you have something real
# to explain in review).
missing_idx = np.random.choice(df.index, size=20, replace=False)
df.loc[missing_idx, "bathrooms"] = np.nan

missing_idx2 = np.random.choice(df.index, size=15, replace=False)
df.loc[missing_idx2, "garage"] = np.nan

df.to_csv("data/housing.csv", index=False)
print(f"Saved {len(df)} rows to data/housing.csv")
print(df.head())
