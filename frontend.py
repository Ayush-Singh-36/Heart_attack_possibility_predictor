# importing all the libraries and dependencies
import streamlit as st
import requests

# page configuration
st.set_page_config(
    page_title = "Heart Attack Risk Predictor",
    page_icon = "♥️",
    layout = "centered"
)

# main title and description
st.title("♥️ Heart Attack Possibility Predictor")
st.markdown(
    """
this application uses a calibrated **Support Vector Machine (SVM)** classifier to evaluate the probability of heart attack risk based on clinical parameters.
please fill in the patient meterics below
"""
)
st.write("---")

# Form layout for user inputs
# we use st.form so the app doesn't refresh on every single click
with st.form(key = "prediction_form"):
    st.subheader("Patient Clinical Data")

    # create two columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value = 1, max_value = 120, value = 54, step = 1)
        sex = st.selectbox("Sex", options = [0, 1], format_func = lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type (cp)", options = [0, 1, 2, 3], format_func = lambda x: f"Type {x}")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value = 50, max_value = 250, value = 125)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value = 100, max_value = 600, value = 273)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options = [0, 1], format_func = lambda x: "True" if x == 1 else "False")

    with col2: 
        restecg = st.selectbox("Resting ECG Results", options = [0, 1, 2])
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value = 60, max_value = 250, value = 152)
        exang = st.selectbox("Exercise Induced Angina", options = [0, 1], format_func = lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value = 0.0, max_value = 10.0, value = 0.5, step = 0.1)
        slope = st.selectbox("Slope of peak Exercise ST Segment", options = [0, 1, 2])
        ca = st.selectbox("Major Vessels Coloured by Fluoroscopy (ca)", options = [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (thal)", options = [0, 1, 2, 3])

    st.write("")
    submit_button = st.form_submit_button(label = "Predict Risk Possibility")

# Handling Form Submission & Connecting to FastAPI Backend
if submit_button:
    # Construct the JSON payload exactly matching your FastAPI Pydantic schema
    payload = {
        "age": int(age),
        "sex": int(sex),
        "cp": int(cp),
        "trestbps": int(trestbps),
        "chol": int(chol),
        "fbs": int(fbs),
        "restecg": int(restecg),
        "thalach": int(thalach),
        "exang": int(exang),
        "oldpeak": float(oldpeak),
        "slope": int(slope),
        "ca": int(ca),
        "thal": int(thal)
    }

    # URL pointing to your active FastAPI /predict endpoint
    FASTAPI_URL = "http://127.0.0.1:8000/predict"
    try:
        with st.spinner("Analyzing data via SVM Model..."):
            # Send the POST request to the backend API
            response = requests.post(FASTAPI_URL, json = payload)

        if response.status_code == 200:
            result = response.json()

            # Extract fields from FastAPI response
            prediction_label = result["prediction_label"]
            confidence_score = result["confidence_score"]

            st.write("---")
            st.subheader("Model Assessment Results")

            # Display metrics dynamically based on risk level
            if result["prediction_code"] == 1:
                st.error(f"⚠️ Assessment : **{prediction_label}**")
                st.metric(label = "Risk Probability Confidence", value = f"{confidence_score}%")
            else:
                st.success(f"✅ Assessment: **{prediction_label}**")
                st.metric(label = "Risk Probability confidence", value = f"{100 - confidence_score}% (Confidence pf Low Risk)")

        else:
            st.error(f"Error form API Server (Status code: {response.status_code}): {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("🚨 Could not connect to FastAPI server. Please make sure your Uvicorn backend server is running on http://127.0.0.1:8000")