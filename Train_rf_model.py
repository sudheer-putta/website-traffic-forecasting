#Train_rf_model.py
# ===============================
# Fraud Detection Model Training Script
# ===============================
# This script loads the fraud detection dataset, preprocesses it,
# trains a Random Forest model, and saves the model and scaler for later use.
#
# Follow the TODOs and fill in the blanks (____) to complete the script!

# 1. Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import os

# 2. Define file paths
data_filepath = "../Data/Fraud_Analysis_Dataset.csv"
model_filepath = "../Models/random_forest_model_v2.joblib"
scaler_filepath = "../Models/scaler.joblib"

# Create Models directory if it doesn't exist
os.makedirs('../Models', exist_ok=True)

# 3. Load the dataset
try:
    fraud_data = pd.read_csv(data_filepath)
    print("Dataset loaded successfully!")
    print(f"Shape of data: {fraud_data.shape}")
except FileNotFoundError:
    print(f"Error: Could not find the data file at {data_filepath}")
    print("Please ensure the data file exists in the correct location")
    exit(1)

# 4. Data preprocessing
# Drop unnecessary columns
fraud_data = fraud_data.drop(['nameOrig', 'nameDest'], axis=1)

# One-hot encode the 'type' column
fraud_data = pd.get_dummies(fraud_data, columns=['type'], drop_first=True)

# 5. Define features and target
X = fraud_data.drop('isFraud', axis=1)
y = fraud_data['isFraud']

# 6. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train-test split completed.")

# 7. Feature scaling
numerical_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])
print("Feature scaling completed.")

# Save the scaler
dump(scaler, scaler_filepath)
print(f"Scaler saved at: {scaler_filepath}")

# 8. Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training completed.")

# 9. Save the trained model
dump(model, model_filepath)
print(f"Trained model saved at: {model_filepath}")

# ===============================
# End of Script
# ===============================
