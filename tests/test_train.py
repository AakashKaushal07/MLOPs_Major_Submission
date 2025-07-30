import pytest
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.utils import load_data
from src.train import main as train_model

# Define a pytest fixture to train the model once and reuse it in tests
@pytest.fixture(scope="module")
def trained_model_and_data():
    """
    Fixture to run the training script and load the resulting model and test data.
    This runs once per test module.
    """
    
    # run training of the model
    train_model()

    model = joblib.load("trained_models/model.joblib")

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
    """Validate model creation (must be a LinearRegression instance)."""
    model, _, _ = trained_model_and_data
    assert isinstance(model, LinearRegression)

def test_model_trained(trained_model_and_data):
    """Check if the model was trained by verifying the 'coef_' attribute exists."""
    model, _, _ = trained_model_and_data
    assert hasattr(model, "coef_")

def test_r2_score_threshold(trained_model_and_data):
    """Ensure R^2 score exceeds a minimum performance threshold."""
    model, X_test, y_test = trained_model_and_data
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    assert score > 0.4