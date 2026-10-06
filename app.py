
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("models/random_forest.pkl")

st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Medical Insurance Cost Predictor")
st.write("Enter your details to estimate medical insurance charges.")

# User inputs
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", 0, 10, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

if st.button("Predict Insurance Charges"):
    # Match the feature encoding used during training
    input_data = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": int(sex == "male"),
        "smoker_yes": int(smoker == "yes"),
        "region_northwest": int(region == "northwest"),
        "region_southeast": int(region == "southeast"),
        "region_southwest": int(region == "southwest")
    }])

    # Ensure the exact training column order
    input_data = input_data[model.feature_names_in_]

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Insurance Charges: ${prediction:,.2f}")
    st.caption(
        "This is a machine-learning estimate, not an official insurance quote."
    )
