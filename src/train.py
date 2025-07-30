import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from utils import load_data

os.makedirs("trained_models", exist_ok=True)

X_train, X_test, y_train, y_test = load_data()

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model: {model.__class__.__name__}")
print(f"Loss (RMSE): {rmse:.3f}")
print(f"R^2 Score: {r2:.3f}")

model_path = os.path.join(".\trained_models", "model.joblib")
joblib.dump(model, model_path)
print(f"Model saved to {model_path} :)")