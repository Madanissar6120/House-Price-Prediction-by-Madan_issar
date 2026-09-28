# 🏠 House Price Predictor

A simple end-to-end machine learning project that predicts house prices from
property features (area, bedrooms, location, etc.), with a Streamlit web UI.

Built as an AIML mini project. This README doubles as a study sheet for
project review — read it end to end before your review.

---

## 1. Project Workflow (the big picture)

```
data/generate_data.py  -->  data/housing.csv
                                   |
                                   v
train_model.py  --(loads csv, cleans, engineers features,
                    trains + evaluates models)-->  model/*.pkl
                                   |
                                   v
app.py  --(loads saved model, takes user input via Streamlit,
            outputs predicted price)
```

Three scripts, three jobs:
- `data/generate_data.py` — creates the dataset.
- `train_model.py` — the actual ML pipeline (preprocessing → training → evaluation → saving).
- `app.py` — the Streamlit UI that loads the *already trained* model and serves predictions. It does not train anything itself.

---

## 2. The Dataset — and why it's synthetic

Real public housing datasets are either:
- Deprecated for ethical reasons (the old "Boston Housing" dataset, which
  used a biased feature), or
- Only available as a download from Kaggle/UCI, which adds a dependency
  on internet access and an external account.

So this project **generates its own dataset** (`data/generate_data.py`)
with `numpy`, using a fixed random seed (`np.random.seed(42)`) for
reproducibility. Price is built as a weighted combination of the features
(bigger area → higher price, older house → lower price, better location
score → higher price) **plus random noise**, so the relationships are
realistic but not perfectly clean — similar to real-world data.

This is a legitimate, defensible choice for a mini project: it's fully
transparent, reproducible, and lets you *know* the ground-truth
relationships, which makes it easy to sanity-check the trained model
(e.g. "does increasing area_sqft increase the predicted price? It should.").

**Features (9 inputs):**
| Feature | Meaning |
|---|---|
| area_sqft | House area in square feet |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| stories | Number of floors |
| age_years | Age of the house in years |
| garage | Number of garage spaces |
| distance_to_city_km | Distance from city center |
| location_score | Locality rating, 1 (poor) – 10 (prime) |
| total_rooms | Engineered = bedrooms + bathrooms |

**Target:** `price`

The generator also injects a small amount of missing data (~1%) into
`bathrooms` and `garage`, so the training pipeline has real preprocessing
to do — this mirrors real-world messy data on a small scale.

---

## 3. Preprocessing

1. **Missing values:** filled using the **median** of each column
   (`fillna(median)`), not the mean — median is robust to outliers, so a
   few unusually large/expensive houses in the data won't distort the
   fill value.
2. **Feature scaling:** `StandardScaler` (zero mean, unit variance).
   This matters most for Linear Regression, which is sensitive to
   features being on very different scales (e.g. `area_sqft` ~ thousands
   vs `location_score` ~ 1–10). Random Forest doesn't strictly need
   scaling, but it's applied to both anyway so there's a single
   consistent pipeline (one scaler, saved once, reused everywhere).

---

## 4. Feature Engineering

`total_rooms = bedrooms + bathrooms` — a simple combined feature that
often correlates strongly with price.

**Deliberately NOT engineered:** `price_per_sqft`. This would divide the
*target* (price) into a feature, which leaks the answer into the input —
a classic mistake called **data leakage**. Be ready to explain this if asked.

---

## 5. Model Selection

Two models are trained and compared:

| Model | Why considered |
|---|---|
| **Linear Regression** | Simple, interpretable baseline. Coefficients directly show each feature's effect on price. |
| **Random Forest Regressor** | Can capture non-linear relationships and feature interactions without manual engineering. |

Both are evaluated on a held-out 20% test set using:
- **RMSE** (Root Mean Squared Error) — penalizes large errors more heavily; in the same units as price (₹).
- **MAE** (Mean Absolute Error) — average absolute prediction error, easier to interpret.
- **R²** — proportion of variance in price explained by the model (closer to 1 is better).

The model with the **lower RMSE** on the test set is automatically
selected and saved. (On this synthetic dataset, Linear Regression
typically wins slightly, because the data was generated from a linear
formula — be ready to explain *why* that makes sense.)

**Alternatives not chosen, and why:**
- *Polynomial Regression* — risk of overfitting with only 9 features and a fairly linear underlying relationship.
- *Gradient Boosting (XGBoost)* — more powerful but heavier dependency and harder to explain simply; overkill for this dataset size.
- *Neural network* — unnecessary complexity for a small tabular regression problem with a mostly-linear signal.

---

## 6. Project Structure

```
house-price-predictor/
├── app.py                     # Streamlit UI
├── train_model.py             # Full training pipeline
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── generate_data.py       # Creates the synthetic dataset
│   └── housing.csv            # Generated dataset (2000 rows)
└── model/
    ├── house_price_model.pkl  # Trained model (best of LR vs RF)
    ├── scaler.pkl             # Fitted StandardScaler
    ├── feature_columns.pkl    # Exact feature order the model expects
    └── model_name.pkl         # Name of the selected model
```

---

## 7. How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/house-price-predictor.git
cd house-price-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Re)generate the dataset — optional, housing.csv is already included
python data/generate_data.py

# 4. Train the model — optional, model/*.pkl is already included
python train_model.py

# 5. Run the app
streamlit run app.py
```

---

## 8. How to Upload to GitHub

```bash
cd house-price-predictor
git init
git add .
git commit -m "Initial commit: house price predictor"
git branch -M main
git remote add origin https://github.com/<your-username>/house-price-predictor.git
git push -u origin main
```

Make sure the repository is set to **Public** on GitHub (Settings → General → Danger Zone → Change visibility, or set it when creating the repo).

> Note: the `model/*.pkl` files ARE committed (not gitignored) on purpose —
> Streamlit Community Cloud only runs `app.py`, not `train_model.py`, so the
> trained model needs to already be in the repo for the deployed app to work.

---

## 9. How to Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**.
3. Select your `house-price-predictor` repository, branch `main`, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install everything from `requirements.txt` automatically.
5. Once deployed, you'll get a public URL like `https://<your-app-name>.streamlit.app` — this is what you submit/show in review.

---

## 10. Results (fill in after you run `train_model.py`)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | ~24,000 | ~19,000 | ~0.92 |
| Random Forest | ~29,000 | ~23,000 | ~0.88 |
| **Selected model** | | | |

(Your exact numbers may vary slightly if you regenerate the dataset, since
noise is randomized — though the fixed seed means they should be close to
identical each run.)

---

## 11. Possible Follow-up Questions to Prepare For

- Why median imputation instead of mean, or dropping rows?
- Why StandardScaler instead of MinMaxScaler?
- Why is `total_rooms` engineered but `price_per_sqft` is not?
- What is data leakage, and where could it have happened here?
- Why RMSE *and* MAE — what's the difference?
- Why did Linear Regression outperform Random Forest here (hint: the underlying data was generated from a roughly linear formula)?
- What would you change if this were real-world data instead of synthetic?
  (e.g. handle outliers more carefully, use cross-validation instead of a single train/test split, try more models, tune hyperparameters with GridSearchCV.)
- Walk through `app.py`: why is `@st.cache_resource` used? Why must `feature_columns` be applied in the exact same order at prediction time as during training?
