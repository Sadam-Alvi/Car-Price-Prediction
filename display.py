import streamlit as st
import pandas as pd
import joblib
import numpy as np
import gdown
import os


# ----------------------------
# 1. Load model, preprocessor & data
# ----------------------------



@st.cache_resource
def load_artifacts():
    model_path = "model.pkl"
    preprocessor_path = "preprocessor.pkl"

    # === Replace these with your own File IDs ===
    model_file_id = "1tPo5g7nu1jYT0gzTvICW0XxvIeSlKIuw"          # ← put your model file ID here
    preprocessor_file_id = "16fg3ckDjUwfL_D2T6TAgC2nYxCwvVBzd"  # ← put your preprocessor file ID here

    # Download only if not already present
    if not os.path.exists(model_path):
        gdown.download(f"https://drive.google.com/uc?id={model_file_id}", model_path, quiet=False)

    if not os.path.exists(preprocessor_path):
        gdown.download(f"https://drive.google.com/uc?id={preprocessor_file_id}", preprocessor_path, quiet=False)

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor

model, preprocessor = load_artifacts()
df = pd.read_csv("autos.csv.zip")  

df['notRepairedDamage'] = df['notRepairedDamage']

# ----------------------------
# 2. Column definitions
# ----------------------------
cat_cols = [
    'abtest',
    'vehicleType',
    'gearbox',
    'model',
    'fuelType',
    'brand',
    'notRepairedDamage'
]

num_cols = ['kilometer', 'yearOfRegistration']

# ----------------------------
# 3. Streamlit UI
# ----------------------------
st.title("Used Car Price Prediction")
st.write("Enter the car details below:")

user_input = {}

# --- Categorical inputs ---
st.subheader("Categorical Features")
for col in cat_cols:
    unique_vals = df[col].dropna().unique().tolist()
    unique_vals = sorted([str(v) for v in unique_vals])
    
    user_input[col] = st.selectbox(
        label=f"{col}",
        options=unique_vals.capitalize(),
        key=col
    )

# --- Numerical inputs ---
st.subheader("Numerical Features")

user_input['kilometer'] = st.number_input(
    "Kilometer",
    # min_value=int(df['kilometer'].min()),
    # max_value=int(df['kilometer'].max()),
    value=int(df['kilometer'].median()),
    step=1000
)

user_input['yearOfRegistration'] = st.number_input(
    "Year of Registration",
    min_value=int(df['yearOfRegistration'].min()),
    max_value=int(df['yearOfRegistration'].max()),
    value=int(df['yearOfRegistration'].median()),
    step=1
)

# ----------------------------
# 4. Prediction
# ----------------------------
if st.button("Predict Price"):
    # Create DataFrame with the same column order as training
    input_df = pd.DataFrame([user_input])
    
    # Ensure correct column order (important for some preprocessors)
    input_df = input_df[cat_cols + num_cols]
    
    # Transform using the same preprocessor
    input_processed = preprocessor.transform(input_df)
    
    # Predict
    prediction = model.predict(input_processed)[0]
    
    st.success(f"**Predicted Price: € {prediction:,.0f}**")









with st.expander("About the Model", expanded=False):
    st.markdown("""
    **Model:** Random Forest Regressor  

    This model uses an ensemble of decision trees to predict the price of used cars.  
    It was trained on historical car listing data and can capture non-linear relationships  
    between features like brand, mileage, year, fuel type, etc.

    **Key advantages of Random Forest:**
    - Handles both categorical and numerical features well  
    - Robust to outliers and missing values  
    - Provides good predictive performance with relatively little tuning  
    """)