import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('german_credit_model.joblib')

st.title("German Credit Risk Predictor")

st.markdown("""
This app predicts whether a loan applicant is a **Good** or **Bad** credit risk.
Please fill in the details below.
""")

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", [
        'skilled',
        'unskilled and resident',
        'highly qualified',
        'unskilled and non-resident'
    ])
    housing = st.selectbox("Housing", ['own', 'rent', 'free'])
    saving_accounts = st.selectbox("Saving accounts", [
        'unknown/ no savings account',
        '< 100 DM',
        '100 <= ... < 500 DM',
        '500 <= ... < 1000 DM',
        '>= 1000 DM'
    ])

with col2:
    credit_amount = st.number_input("Credit amount (DM)", min_value=0, value=1000)
    duration = st.number_input("Duration (months)", min_value=1, max_value=100, value=12)
    purpose = st.selectbox("Purpose", [
        'radio/television',
        'education',
        'furniture/equipment',
        'car (new)',
        'car (used)',
        'business',
        'domestic appliances',
        'repairs',
        'others',
        'retraining'
    ])

# Missing inputs that are required by the model but not in user requirements
# We will add them as optional or default, or just add them to the UI.
# The prompt specified specific fields, but the model needs these to function.
st.markdown("---")
st.markdown("### Additional Required Information")
col3, col4 = st.columns(2)
with col3:
    checking_account = st.selectbox("Checking account", [
        '< 0 DM',
        '0 <= ... < 200 DM',
        '>= 200 DM / salary assignments for at least 1 year',
        'no checking account'
    ])
with col4:
    sex = st.selectbox("Sex", ['male', 'female'])

# Prediction
if st.button("Predict Risk"):
    # Create DataFrame from input
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Job': [job],
        'Housing': [housing],
        'Saving accounts': [saving_accounts],
        'Checking account': [checking_account],
        'Credit amount': [credit_amount],
        'Duration': [duration],
        'Purpose': [purpose]
    })

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # Probability of 'Bad'

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"Risk: **BAD** (Probability: {probability:.2%})")
        st.write("This application is flagged as high risk.")
    else:
        st.success(f"Risk: **GOOD** (Probability: {1-probability:.2%})")
        st.write("This application appears to be low risk.")

    # Show input data for debugging/confirmation
    with st.expander("See Input Data"):
        st.write(input_data)
