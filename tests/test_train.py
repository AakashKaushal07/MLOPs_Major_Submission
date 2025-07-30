import pytest
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.utils import load_data

@pytest.fixture(scope="module")
def trained_model_and_data():
    os.system("python ./src/train.py")
    model = joblib.load("./trained_models/model.joblib")
    _, X_test, _, y_test = load_data()

    return model, X_test, y_test

def test_data_loading():
    """Unit test for dataset loading."""
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

def test_model_creation(trained_model_and_data):
    """Validate model creation (LinearRegression instance)."""
    model, _, _ = trained_model_and_data
    assert isinstance(model, LinearRegression)

def test_model_trained( trained_model_and_data):
    """Check if the model was trained (e.g., coef_ exists)."""
    model, _, _ = trained_model_and_data
    assert hasattr(model, "coef_")

def test_r2_score_threshold(trained_model_and_data):
    """Ensure R^2 score exceeds a minimum threshold."""
    model, X_test, y_test = trained_model_and_data
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    assert score > 0.4