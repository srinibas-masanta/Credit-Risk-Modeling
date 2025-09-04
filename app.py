import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("random_forest_credit_model.pkl")
encoders = {
    col: joblib.load(f"{col}_encoder.pkl")
    for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
}

# App Title
st.title("Credit Risk Prediction App")
st.write("Enter applicant information to predict if the credit risk is good or bad")

# Input fields
age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ["own", "free", "rent"])
saving_accounts = st.selectbox("Saving accounts", ["little", "moderate", "quite rich", "rich", "unknown"])
checking_account = st.selectbox("Checking account", ["little", "moderate", "rich", "unknown"])
credit_amount = st.number_input("Credit Amount", min_value=0, value=100)
duration = st.number_input("Duration (months)", min_value=1, value=12)

# Encode categorical features safely
try:
    sex_enc = encoders["Sex"].transform([sex])[0]
    housing_enc = encoders["Housing"].transform([housing])[0]
    saving_enc = encoders["Saving accounts"].transform([saving_accounts])[0]
    checking_enc = encoders["Checking account"].transform([checking_account])[0]
except Exception as e:
    st.error(f"Encoding failed: {e}")
    st.stop()

# Create input dataframe
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [sex_enc],
    "Job": [job],
    "Housing": [housing_enc],
    "Saving accounts": [saving_enc],
    "Checking account": [checking_enc],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

# Predict
if st.button("Predict Risk"):
    try:
        pred = model.predict(input_df)[0]
        
        # Handle numeric or string predictions
        if str(pred).lower() in ["1", "good"]:
            st.success("✅ The predicted credit risk is: **GOOD**")
        else:
            st.error("⚠️ The predicted credit risk is: **BAD**")
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")
