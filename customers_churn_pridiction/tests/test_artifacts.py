import pytest
import joblib
from pathlib import Path

MODEL_DIR = Path("models")
REQUIRED = ["churn_model.joblib", "label_encoder.joblib", "scaler.joblib", "onehot_encoder.joblib"]

@pytest.mark.order(3)
def test_artifacts_exist():
    """Ensure required model artifacts exist."""
    for file in REQUIRED:
        path = MODEL_DIR / file
        assert path.exists(), f"{file} missing in models directory!"

@pytest.mark.order(4)
def test_artifacts_loadable():
    """Ensure all artifacts can be loaded properly."""
    for file in REQUIRED:
        path = MODEL_DIR / file
        try:
            obj = joblib.load(path)
            assert obj is not None
        except Exception as e:
            pytest.fail(f"Failed to load {file}: {e}")
