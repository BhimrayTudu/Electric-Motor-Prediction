import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('Random_model.pkl')

# Define feature names (should match the model's expected input)
feature_names = ['ambient', 'coolant', 'u_d', 'u_q', 'torque', 'i_d', 'i_q', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding']

# Title of the application
st.title("Electric Motor Prediction")

# User input section
st.sidebar.header("Input Features")

# Dynamically generate input fields
input_values = []
for feature in feature_names:
    value = st.sidebar.number_input(f"Enter {feature}", value=25.0)  # Adjust default value
    input_values.append(value)

# Convert input values to NumPy array
input_data = np.array([input_values]).reshape(1, -1)

# Prediction
if st.sidebar.button("Predict"):
    if input_data.shape[1] != model.n_features_in_:
        st.sidebar.error(f"Feature mismatch: Model expects {model.n_features_in_} features, but received {input_data.shape[1]}")
    else:
        prediction = model.predict(input_data)
        st.sidebar.success(f"Predicted Value: {prediction[0]:.2f}")
