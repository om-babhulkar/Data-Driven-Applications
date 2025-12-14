import sys
import os
from pathlib import Path

try:
    src_path = str(Path(__file__).parent / "src")
    if src_path not in sys.path:
        sys.path.append(src_path)
    
    from src.ml_pipeline.preprocess import preprocess_and_split
    from src.ml_pipeline.train_model import train_and_save
    from src.ml_pipeline.test import evaluate
    from src.utils.logger import log_info, log_error, get_logger
    
    main_logger = get_logger("main_pipeline", Path(__file__).parent / "logs" / "main.log")
    log_info("Main pipeline execution started.", logger=main_logger)

except ImportError as e:
    print(f"Error: Failed to import pipeline modules. Make sure 'src' directory is correct.")
    print(f"Details: {e}")
    sys.exit(1)


def run_pipeline():
    """
    Executes the full ML pipeline: preprocess, train, and evaluate.
    """
    try:
        log_info("--- Step 1: Preprocessing Data ---", logger=main_logger)
        preprocess_and_split()
        log_info("--- Step 1: Preprocessing COMPLETE ---", logger=main_logger)
    except Exception as e:
        log_error(f"Preprocessing failed: {e}", logger=main_logger)
        return  

    try:
        log_info("--- Step 2: Training Model ---", logger=main_logger)
        train_and_save()
        log_info("--- Step 2: Training COMPLETE ---", logger=main_logger)
    except Exception as e:
        log_error(f"Model training failed: {e}", logger=main_logger)
        return  

    try:
        log_info("--- Step 3: Evaluating Model ---", logger=main_logger)
        evaluate()
        log_info("--- Step 3: Evaluation COMPLETE ---", logger=main_logger)
    except Exception as e:
        log_error(f"Model evaluation failed: {e}", logger=main_logger)
        return

    log_info("ML Pipeline executed successfully.", logger=main_logger)

if __name__ == "__main__":
    run_pipeline()
