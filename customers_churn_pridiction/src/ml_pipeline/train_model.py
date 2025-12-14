from sklearn.ensemble import RandomForestClassifier
from src.utils.logger import log_info, log_error, log_warning, train_logger
from src.utils.helper import safe_save_joblib, safe_load_joblib
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

def train_and_save():
    log_info("--- Model Training Started ---", logger=train_logger)


    try:
        data_path = MODELS_DIR / "data_splits.joblib"
        X_train, X_test, y_train, y_test = safe_load_joblib(data_path)
        log_info(f"Loaded processed data splits from {data_path}", logger=train_logger)
    except Exception as e:
        log_error(f"Failed to load data splits: {e}. Run preprocess.py first.", logger=train_logger)
        raise e
    
    try:
        log_info("Training RandomForestClassifier...", logger=train_logger)
        model = RandomForestClassifier( random_state=42)
        model.fit(X_train, y_train)
        log_info("Model training complete.", logger=train_logger)
    except Exception as e:
        log_error(f"Error during model training: {e}", logger=train_logger)
        raise e

    try:
        model_path = MODELS_DIR / "churn_model.joblib"
        safe_save_joblib(model, model_path)
        log_info(f"Model saved to {model_path}", logger=train_logger)
    except Exception as e:
        log_error(f"Error saving model: {e}", logger=train_logger)
        raise e
    
    log_info("--- Model Training Finished ---", logger=train_logger)

if __name__ == "__main__":
    train_and_save()

