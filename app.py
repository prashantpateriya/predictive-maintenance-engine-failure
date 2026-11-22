import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(page_title="Automobile Engine Predictive Maintenance", layout="centered")

st.title("🚗 Automobile Engine Predictive Maintenance Monitor")
st.write("""
This Streamlit application leverages our optimized Machine Learning pipeline
to predict engine failures and determine if proactive maintenance is required.
""")

# Path to the serialized model file
model_path = "best_predictive_maintenance_model.pkl"
if not os.path.exists(model_path):
    # Fallback path for deployment
    model_path = "predictive_maintenance_project/models/best_predictive_maintenance_model.pkl"

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

model = load_model(model_path)

if model is None:
    st.error("⚠️ Trained model artifact was not found in the repository paths. Please verify model serialization stage.")
else:
    st.success("✅ Predictive Maintenance Engine Model loaded successfully!")

    st.header("🔌 Real-Time Telemetry Sensor Inputs")

    col1, col2 = st.columns(2)

    with col1:
        engine_rpm = st.number_input("Engine RPM", min_value=0, max_value=10000, value=3000, step=100)
        lub_oil_pressure = st.number_input("Lubrication Oil Pressure (bar)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
        fuel_pressure = st.number_input("Fuel Pressure (bar)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)

    with col2:
        coolant_pressure = st.number_input("Coolant Pressure (bar)", min_value=0.0, max_value=10.0, value=2.2, step=0.1)
        lub_oil_temp = st.number_input("Lubrication Oil Temperature (°C)", min_value=0.0, max_value=150.0, value=85.0, step=1.0)
        coolant_temp = st.number_input("Coolant Temperature (°C)", min_value=0.0, max_value=130.0, value=78.0, step=1.0)

    # Generate prediction on request
    if st.button("Run Maintenance Diagnostics", type="primary"):
        # Ensure identical feature naming and order as used during training
        input_data = pd.DataFrame({
            'Engine rpm': [engine_rpm],
            'Lub oil pressure': [lub_oil_pressure],
            'Fuel pressure': [fuel_pressure],
            'Coolant pressure': [coolant_pressure],
            'lub oil temp': [lub_oil_temp],
            'Coolant temp': [coolant_temp]
        })

        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        st.subheader("📊 Maintenance Diagnostic Results")

        if prediction == 1:
            st.error(f"🚨 ALERT: Maintenance Required (Risk Probability: {prediction_proba[1]:.2%})")
            st.warning("⚠️ Actionable Advice: Schedule immediate technician diagnostics. Check lubrication levels, coolant flow, and fuel injector status to prevent unexpected breakdown.")
        else:
            st.success(f"✨ System Healthy: Normally Operating (Safety Confidence: {prediction_proba[0]:.2%})")
            st.info("ℹ️ Actionable Advice: Engine operating metrics are within normal bounds. Proceed with normal scheduling protocols.")
