import pytest
from pathlib import Path
from src.ml_pipeline.preprocess import preprocess_and_split
from src.ml_pipeline.train_model import train_and_save
from src.ml_pipeline.test import evaluate
import joblib
import os

@pytest.mark.order(1)
def test_preprocess_and_train(tmp_path):
    """✅ Test preprocessing and training pipeline end-to-end."""

    preprocess_and_split()

    assert Path("models/data_splits.joblib").exists(), "❌ Data splits file missing!"
    assert Path("models/scaler.joblib").exists(), "❌ Scaler file missing!"
    assert Path("models/label_encoder.joblib").exists(), "❌ Label encoder missing!"
    assert Path("models/onehot_encoder.joblib").exists(), "❌ OneHotEncoder missing!"

    train_and_save()

    model_path = Path("models/churn_model.joblib")
    assert model_path.exists(), "❌ Model not saved!"

    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "❌ Model object invalid (no predict method)."


@pytest.mark.order(2)
def test_model_evaluation():
    """✅ Test model evaluation and report content."""

    report = evaluate()
    assert report is not None, "❌ Evaluation report missing!"
    assert "accuracy" in report, "❌ Accuracy metric not found!"
    assert report["accuracy"] > 0.7, f"⚠️ Model accuracy too low ({report['accuracy']:.2%})"
