# src/predict.py

import joblib
import os
import numpy as np
from utils import load_data

print("Running prediction script...")

model_path = os.path.join("trained_models", "model.joblib")
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    print("Please run the training script (src/train.py) first.")
    exit()

_, X_test, _, y_test = load_data()
print("Test data loaded.")

sample_indices = np.random.choice(X_test.shape[0], 5, replace=False)
sample_data = X_test[sample_indices]
sample_actual = y_test[sample_indices]

sample_predictions = model.predict(sample_data)
print("Prediction complete on sample data.")

print("\n--- Sample Predictions ---")
for i in range(len(sample_indices)):
    print(f"Sample #{i+1}:")
    print(f"  - Actual Value:    {sample_actual[i]:.4f}")
    print(f"  - Predicted Value: {sample_predictions[i]:.4f}\n")