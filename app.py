import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('klasifikasi_obesitas_svm.pkl')

# Load the scaler
scaler = StandardScaler()
scaler.fit(X_train_smote)

# Define a function to predict the BMI category
def predict_bmi_category(height, weight, gender):
  # Preprocess the input data
  data = [[height, weight, gender]]
  scaled_data = scaler.transform(data)

  # Predict the BMI category
  prediction = model.predict(scaled_data)[0]

  # Return the predicted category
  return prediction

# Example usage
height = 170
weight = 60
gender = 0  # Male

predicted_category = predict_bmi_category(height, weight, gender)
print(f"Predicted BMI category: {predicted_category}")
