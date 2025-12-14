import sys
from pathlib import Path
import traceback

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.logger import log_info, log_error, main_logger
from src.ml_pipelines.preprocess import preprocess_split
from src.ml_pipelines.train import train_and_save
from src.ml_pipelines.test import evaluate_model

def run_pipeline():
    """
    Executes the full preprocessing, training, and evaluation pipeline.
    """
    log_info("==========  STARTING FULL PIPELINE  ==========", logger=main_logger)
    
    try:
        log_info("--- Starting Step 1: Data Preprocessing ---", logger=main_logger)
        preprocess_split(save_json=True)
        log_info("--- Completed Step 1: Data Preprocessing ---", logger=main_logger)

        log_info("--- Starting Step 2: Model Training ---", logger=main_logger)
        train_metrics = train_and_save() 
        log_info("--- Completed Step 2: Model Training ---", logger=main_logger)
        
        log_info("--- Starting Step 3: Model Evaluation ---", logger=main_logger)
        test_metrics = evaluate_model()
        log_info("--- Completed Step 3: Model Evaluation ---", logger=main_logger)
        
        log_info("\n==========  FULL PIPELINE COMPLETED SUCCESSFULLY  ==========", logger=main_logger)
        log_info(f"Model Path: {train_metrics.get('model_path', 'N/A')}", logger=main_logger)
        log_info(f"Final Test Set Metrics (from test.py):", logger=main_logger)
        log_info(f"  Accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}", logger=main_logger)
        log_info(f"  ROC-AUC: {test_metrics.get('roc_auc', 'N/A'):.4f}", logger=main_logger)

    except Exception as e:
        error_details = traceback.format_exc()
        log_error(f"==========  PIPELINE FAILED  ==========", logger=main_logger)
        log_error(f"Error: {e}", logger=main_logger)
        log_error(f"Traceback:\n{error_details}", logger=main_logger)
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()