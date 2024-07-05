import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('klasifikasi_obesitas_svm.pkl')

# Load the scaler (assuming X_train_smote is defined elsewhere)
scaler = StandardScaler()
# scaler.fit(X_train_smote)  # Uncomment this line if X_train_smote is defined

# Function to predict BMI category
def predict_bmi_category(height, weight, gender):
    # Preprocess the input data
    data = [[height, weight, gender]]
    scaled_data = scaler.transform(data)

    # Predict the BMI category
    prediction = model.predict(scaled_data)[0]

    # Return the predicted category
    return prediction

# Streamlit application title
st.title('BMI Category Prediction')

# Input fields for user
gender = st.selectbox('Select Gender', ('Female', 'Male'))
height = st.number_input('Height (in cm)', min_value=100, max_value=300, value=170)
weight = st.number_input('Weight (in kg)', min_value=20, max_value=300, value=60)

# Convert gender to numeric
gender_num = 0 if gender == 'Female' else 1

# Prediction button
if st.button('Predict BMI Category'):
    # Predict BMI category
    predicted_category = predict_bmi_category(height, weight, gender_num)

    # Display prediction
    bmi_categories = {0: 'Underweight', 1: 'Normal', 2: 'Overweight', 3: 'Obese'}
    st.write(f'Predicted BMI Category: {bmi_categories[predicted_category]}')
