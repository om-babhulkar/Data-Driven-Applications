from pathlib import Path
import sys
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)
from datetime import datetime
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
from src.utils.logger import log_info, log_error, log_warning, train_logger
from src.utils.helper import safe_save_joblib, safe_load_joblib

BASE = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train_and_save():
    log_info("==========  Model Training Pipeline Started ==========", logger=train_logger)

    try:
        data_path = MODELS_DIR / "data_splits.joblib"
        X_train, X_test, y_train, y_test = safe_load_joblib(data_path)
        log_info(f" Loaded processed data splits from: {data_path}", logger=train_logger)
    except Exception as e:
        log_error(f" Failed to load data splits: {e}. Run preprocess.py first.", logger=train_logger)
        raise e

    try:
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        log_info(" Model compiled successfully.", logger=train_logger)

    except Exception as e:
        log_error(f" Failed to compile the model: {e}", logger=train_logger)
        raise e

    try:
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=64,
            callbacks=[early_stop],
            verbose=1
        )
        log_info("Model training completed successfully.", logger=train_logger)

    except Exception as e:
        log_error(f" Failed during model training: {e}", logger=train_logger)
        raise e

    try:
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred_prob)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        log_info(f"Accuracy: {acc:.4f}", logger=train_logger)
        log_info(f"ROC-AUC: {roc:.4f}", logger=train_logger)
        log_info(f"Confusion Matrix:\n{cm}", logger=train_logger)

    except Exception as e:
        log_error(f" Evaluation failed: {e}", logger=train_logger)
        raise e

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = MODELS_DIR / f"ann_model_{timestamp}.keras" 
        metrics_path = MODELS_DIR / f"metrics_{timestamp}.joblib"
        history_path = MODELS_DIR / f"history_{timestamp}.joblib"

        model.save(model_path)
        log_info(f"Model saved at: {model_path}", logger=train_logger)

        safe_save_joblib(history.history, history_path)
        log_info(f"Training history saved at: {history_path}", logger=train_logger)

        metrics = {
            "accuracy": acc,
            "roc_auc": roc,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "timestamp": timestamp,
        }
        safe_save_joblib(metrics, metrics_path)
        log_info(f"Metrics saved at: {metrics_path}", logger=train_logger)

        log_info("All artifacts saved successfully.", logger=train_logger)

    except Exception as e:
        log_error(f" Failed to save artifacts: {e}", logger=train_logger)
        raise e

    log_info("========== Model Training Pipeline Completed Successfully ==========", logger=train_logger)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "accuracy": acc,
        "roc_auc": roc
    }

if __name__ == "__main__":
    try:
        result = train_and_save()
        log_info(f"Final Result: {result}", logger=train_logger)
    except Exception as e:
        log_error(f"Training pipeline failed: {e}", logger=train_logger)
        sys.exit(1)