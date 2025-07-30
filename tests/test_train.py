import pytest
import joblib
import subprocess
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.utils import load_data


@pytest.fixture(scope="module")
def trained_model_and_data():
    subprocess.run(["python", "./src/train.py"], check=True)

    model_path = "trained_models/model.joblib"
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    model = joblib.load(model_path)

    _, X_test, _, y_test = load_data()
    return model, X_test, y_test


def test_data_loading():
    """Unit test for dataset loading."""
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[0] > 0, "X_train is empty"
    assert X_test.shape[0] > 0, "X_test is empty"
    assert len(y_train) > 0, "y_train is empty"
    assert len(y_test) > 0, "y_test is empty"


def test_model_creation(trained_model_and_data):
    """Validate model creation."""
    model, _, _ = trained_model_and_data
    assert isinstance(model, LinearRegression), "Model is not a LinearRegression instance"


def test_model_trained(trained_model_and_data):
    """Check if model has been fitted (has 'coef_' attribute)."""
    model, _, _ = trained_model_and_data
    assert hasattr(model, "coef_"), "Model doesn't have coef_ attribute"


def test_r2_score_threshold(trained_model_and_data):
    """Ensure R² score is above threshold."""
    model, X_test, y_test = trained_model_and_data
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    assert score > 0.4, f"R² score too low: {score:.2f}"
