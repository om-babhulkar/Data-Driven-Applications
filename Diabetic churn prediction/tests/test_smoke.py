import sys
from pathlib import Path
import pytest

# Add the root and src directories to the path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(BASE_DIR))
sys.path.append(str(SRC_DIR))

# Test that all main scripts are importable
def test_imports():
    try:
        import app
        import main
        from src.ml_pipelines import preprocess, train, test
        from src.utils import helper, logger
    except ImportError as e:
        pytest.fail(f"Failed to import a module: {e}")

# Test that loggers were created
def test_loggers_exist():
    from src.utils import logger
    assert logger.main_logger is not None
    assert logger.app_logger is not None
    assert logger.preprocess_logger is not None
    assert logger.train_logger is not None
    assert logger.evaluate_logger is not None

# Test that helper functions exist
def test_helper_functions_exist():
    from src.utils import helper
    assert hasattr(helper, 'safe_save_joblib')
    assert hasattr(helper, 'safe_load_joblib')
    assert hasattr(helper, 'find_latest_file')