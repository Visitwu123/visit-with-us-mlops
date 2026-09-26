
import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "tourism_project/model_building/best_model.joblib"

st.set_page_config(
    page_title="Visit With Us - Wellness Package Prediction",
    layout="centered"
)

st.title("Visit With Us")
st.subheader("Wellness Tourism Package Prediction")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found in the repository.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.success("Model loaded successfully.")

st.write(
    "Enter customer details below to predict whether the customer "
    "is likely to purchase the Wellness Tourism Package."
)

st.subheader("Customer Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

type_of_contact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

duration_of_pitch = st.number_input(
    "Duration of Pitch",
    min_value=1,
    max_value=60,
    value=15
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

number_of_person_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2
)

number_of_followups = st.number_input(
    "Number of Followups",
    min_value=0,
    max_value=10,
    value=3
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)

preferred_property_star = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Married", "Divorced", "Single", "Unmarried"]
)

number_of_trips = st.number_input(
    "Number of Trips",
    min_value=0,
    max_value=30,
    value=3
)

passport = st.selectbox(
    "Passport",
    [0, 1]
)

pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

own_car = st.selectbox(
    "Own Car",
    [0, 1]
)

number_of_children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=1
)

designation = st.selectbox(
    "Designation",
    ["AVP", "Executive", "Manager", "Senior Manager", "VP"]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=1000000,
    value=25000
)

if st.button("Predict Package Purchase"):

    input_data = pd.DataFrame({
        "Age": [age],
        "TypeofContact": [type_of_contact],
        "CityTier": [city_tier],
        "DurationOfPitch": [duration_of_pitch],
        "Occupation": [occupation],
        "Gender": [gender],
        "NumberOfPersonVisiting": [number_of_person_visiting],
        "NumberOfFollowups": [number_of_followups],
        "ProductPitched": [product_pitched],
        "PreferredPropertyStar": [preferred_property_star],
        "MaritalStatus": [marital_status],
        "NumberOfTrips": [number_of_trips],
        "Passport": [passport],
        "PitchSatisfactionScore": [pitch_satisfaction_score],
        "OwnCar": [own_car],
        "NumberOfChildrenVisiting": [number_of_children_visiting],
        "Designation": [designation],
        "MonthlyIncome": [monthly_income]
    })

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction")

    if prediction == 1:
        st.success(
            "Customer is predicted to purchase the Wellness Tourism Package."
        )
    else:
        st.info(
            "Customer is predicted not to purchase the Wellness Tourism Package."
        )

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]

        st.write(
            f"Estimated purchase probability: "
            f"{probability:.2%}"
        )
