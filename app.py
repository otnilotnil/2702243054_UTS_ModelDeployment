import streamlit as st
import pickle
import numpy as np
import xgboost
@st.cache_resource
def load_model():
    with open('finalXGBModel.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('labelEncoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('topFeatureNames.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, encoders, features

model, encoders, features = load_model()

st.title("🏨 Hotel Booking Prediction App")

# 2. Build input form based on features used
user_input = {}

user_input['lead_time'] = st.number_input("Lead Time", 0, 450, 100)

user_input['no_of_special_requests'] = st.slider("Special Requests", 0, 5, 1)

user_input['arrival_year'] = st.selectbox("Arrival Year", [2017, 2018])

user_input['avg_price_per_room'] = st.number_input("Average Price per Room", 0, 550, 100)

market_options = encoders['market_segment_type'].classes_.tolist()
selected_market = st.selectbox("Market Segment Type", market_options)
user_input['market_segment_type'] = encoders['market_segment_type'].transform([selected_market])[0]

if st.button("Predict Booking Status"):
    input_array = np.array([user_input[feature] for feature in features]).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    # Decode the target value if needed
    status = encoders['booking_status'].inverse_transform([prediction])[0]
    st.success(f"📌 Booking Status Prediction: **{status}**")
