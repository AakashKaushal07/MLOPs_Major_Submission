import joblib
import numpy as np
import os
from sklearn.metrics import r2_score
from src.utils import load_data

print("Starting quantization process...")

model_path = os.path.join("./trained_models", "model.joblib")
model = joblib.load(model_path)
print("Trained model loaded.")

coef = model.coef_
intercept = model.intercept_
params = {"coef": coef, "intercept": intercept}
print("Extracted model parameters.")

unquant_path = os.path.join("./trained_models", "unquant_params.joblib")
joblib.dump(params, unquant_path)
print(f"Unquantized parameters saved to {unquant_path}")

all_params = np.concatenate([coef, np.array([intercept])])
min_val = all_params.min()
max_val = all_params.max()

scale = (max_val - min_val) / 255.0
zero_point = -min_val / scale
zero_point = np.round(np.clip(zero_point, 0, 255))

quant_coef = np.round(coef / scale + zero_point).astype(np.uint8)
quant_intercept = np.round(intercept / scale + zero_point).astype(np.uint8)
print("Parameters quantized to uint8.")

quant_params = {
    "quant_coef": quant_coef,
    "quant_intercept": quant_intercept,
    "scale": scale,
    "zero_point": zero_point,
}
quant_path = os.path.join("./trained_models", "quant_params.joblib")
joblib.dump(quant_params, quant_path)
print(f"Quantized parameters saved to {quant_path}")

print("\nVerifying quantization by running inference...")
_, X_test, _, y_test = load_data()

dequant_coef = (quant_coef.astype(np.float32) - zero_point) * scale
dequant_intercept = (quant_intercept.astype(np.float32) - zero_point) * scale

y_pred_quant = np.dot(X_test, dequant_coef) + dequant_intercept
r2_quant = r2_score(y_test, y_pred_quant)

print(f"R^2 score with de-quantized parameters: {r2_quant:.4f}")