import sys
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)
import numpy as np
import warnings

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.logger import log_info, log_error, evaluate_logger
from src.utils.helper import safe_load_joblib, find_latest_file

MODELS_DIR = BASE_DIR / "models"
MODEL_PATTERN = "ann_model_*.keras"

warnings.filterwarnings("ignore", category=FutureWarning)
tf.get_logger().setLevel('ERROR')


def evaluate_model():
    """
    Loads the latest trained model and evaluates it on the test set.
    """
    log_info("========== Model Evaluation Pipeline Started ==========", logger=evaluate_logger)
    metrics = {}

    try:
        data_path = MODELS_DIR / "data_splits.joblib"
        _, X_test, _, y_test = safe_load_joblib(data_path)
        log_info(f"Loaded test data from: {data_path}", logger=evaluate_logger)
    except Exception as e:
        log_error(f"Failed to load data splits: {e}. Run preprocessing first.", logger=evaluate_logger)
        return metrics

    try:
        latest_model_path = find_latest_file(MODELS_DIR, MODEL_PATTERN)
        if not latest_model_path:
            log_error(f"No model file matching '{MODEL_PATTERN}' found in {MODELS_DIR}.", logger=evaluate_logger)
            return metrics
            
        model = tf.keras.models.load_model(latest_model_path)
        log_info(f"Successfully loaded model from: {latest_model_path}", logger=evaluate_logger)
    except Exception as e:
        log_error(f"Failed to load model: {e}", logger=evaluate_logger)
        return metrics

    try:
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred_prob)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"])

        metrics = {"accuracy": acc, "roc_auc": roc}

        log_info("========== Test Set Evaluation Results ==========", logger=evaluate_logger)
        log_info(f"\nAccuracy: {acc:.4f}", logger=evaluate_logger)
        log_info(f"ROC-AUC: {roc:.4f}", logger=evaluate_logger)
        log_info(f"\nConfusion Matrix:\n{cm}", logger=evaluate_logger)
        log_info(f"\nClassification Report:\n{report}", logger=evaluate_logger)
        log_info("=================================================", logger=evaluate_logger)

    except Exception as e:
        log_error(f"Error during model evaluation: {e}", logger=evaluate_logger)
        raise e

    log_info("========== Model Evaluation Pipeline Completed ==========", logger=evaluate_logger)
    return metrics


if __name__ == "__main__":
    evaluate_model()