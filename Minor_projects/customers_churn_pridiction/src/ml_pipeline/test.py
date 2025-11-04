from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
from pathlib import Path
from src.utils.logger import log_info, log_error, log_warning, evaluate_logger
from src.utils.helper import safe_load_joblib, save_to_json
import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE / "models"

def evaluate():
    log_info("--- Model Evaluation Started ---", logger=evaluate_logger)
    proba = None

    try:
        model_path = MODELS_DIR / "churn_model.joblib"
        model = safe_load_joblib(model_path)
        log_info(f"Model loaded from {model_path}", logger=evaluate_logger)
    except Exception as e:
        log_error(f"Failed to load model: {e}", logger=evaluate_logger)
        raise e

    try:
        data_path = MODELS_DIR / "data_splits.joblib"
        _, X_test, _, y_test = safe_load_joblib(data_path)
        log_info(f"Test data loaded from {data_path}", logger=evaluate_logger)
    except Exception as e:
        log_error(f"Failed to load test data: {e}", logger=evaluate_logger)
        raise e

    try:
        pred = model.predict(X_test)
        log_info("Prediction completed on test set.", logger=evaluate_logger)
    except Exception as e:
        log_error(f"Prediction failed: {e}", logger=evaluate_logger)
        raise e

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1] 
            log_info("Probability prediction done.", logger=evaluate_logger)
        else:
            log_warning("Model does not support 'predict_proba'. Skipping AUC.", logger=evaluate_logger)
    except Exception as e:
        log_error(f"Failed to calculate probabilities: {e}", logger=evaluate_logger)

    reports = {}
    
    try:
        log_info("Calculating metrics...", logger=evaluate_logger)
        reports['accuracy'] = accuracy_score(y_test, pred)
        reports['classification_report'] = classification_report(y_test, pred, output_dict=True)
        reports['confusion_matrix'] = confusion_matrix(y_test, pred).tolist() # Convert to list for JSON
        
        if proba is not None:
            reports['roc_auc_score'] = roc_auc_score(y_test, proba)
            
        log_info(f"Accuracy: {reports['accuracy']:.4f}")
        if 'roc_auc_score' in reports:
            log_info(f"ROC AUC: {reports['roc_auc_score']:.4f}")

    except Exception as e:
        log_error(f"Failed to generate reports: {e}", logger=evaluate_logger)
        raise e
        
    try:
        report_path = MODELS_DIR / "evaluation_report.json"
        save_to_json(reports, report_path)
        log_info(f"Evaluation report saved to {report_path}", logger=evaluate_logger)
    except Exception as e:
        log_error(f"Failed to save evaluation report: {e}", logger=evaluate_logger)
        raise e
        
    log_info("--- Model Evaluation Finished ---", logger=evaluate_logger)
    
    return reports


if __name__ == "__main__":
    evaluate()
