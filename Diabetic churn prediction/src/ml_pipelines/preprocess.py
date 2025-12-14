from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from src.utils.logger import log_info, log_error, log_warning, preprocess_logger
from src.utils.helper import safe_save_joblib, save_to_json

BASE = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE / "dataset" 
DATA_PATH = DATASET_DIR / "diabetes_dataset.csv"
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLS = [
    "education_level",
    "income_level",
    "employment_status",
    "diabetes_stage",
]
TARGET_COL = "diagnosed_diabetes"

report = {}


def preprocess_split(data_path: Path = DATA_PATH, save_json: bool = True, test_size: float = 0.25, random_state: int = 42):
    MODELS_DIR.mkdir(exist_ok=True)

    try:
        df = pd.read_csv(data_path)
        log_info(
            f"Dataset loaded successfully from {data_path}: {df.shape[0]} rows × {df.shape[1]} columns",
            logger=preprocess_logger
        )
    except FileNotFoundError:
        log_error(f"FATAL: Dataset not found at {data_path}.", logger=preprocess_logger)
        raise
    except Exception as e:
        log_error(f"Failed to load dataset: {e}", logger=preprocess_logger)
        raise e

    try:
        cols_to_drop = [c for c in DROP_COLS if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            log_info(f"Dropped columns: {cols_to_drop}", logger=preprocess_logger)
    except Exception as e:
        log_error(f"Error dropping columns: {e}", logger=preprocess_logger)
        raise e

    try:
        if df.isnull().sum().any():
            log_warning("NaN values detected. Filling with median (numeric) or mode (categorical).", logger=preprocess_logger)
            for col in df.columns:
                if df[col].dtype == "object" or df[col].dtype.name == "category":
                    if not df[col].mode().empty:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    else:
                        df[col].fillna("", inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
            log_info("Missing values filled.", logger=preprocess_logger)
    except Exception as e:
        log_error(f"Error during NaN handling: {e}", logger=preprocess_logger)
        raise e

    if TARGET_COL not in df.columns:
        log_error(f"FATAL: Target column '{TARGET_COL}' not found in dataset.", logger=preprocess_logger)
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    try:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if TARGET_COL in num_cols:
            num_cols.remove(TARGET_COL)
            
        log_info(f"Identified {len(cat_cols)} categorical and {len(num_cols)} numeric columns.", logger=preprocess_logger)
        

        safe_save_joblib(cat_cols, MODELS_DIR / "categorical_columns.joblib")
        safe_save_joblib(num_cols, MODELS_DIR / "numerical_columns.joblib")
        log_info("Saved categorical and numerical column lists for API.", logger=preprocess_logger)

    except Exception as e:
        log_error(f"Error identifying column types: {e}", logger=preprocess_logger)
        raise e

    try:
        if cat_cols:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded_array = encoder.fit_transform(df[cat_cols])
            encoded_cols = encoder.get_feature_names_out(cat_cols)
            encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
            df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)
            safe_save_joblib(encoder, MODELS_DIR / "onehot_encoder.joblib")
            log_info("OneHotEncoder applied and saved.", logger=preprocess_logger)
        else:
            log_info("No categorical columns found. Skipping OneHotEncoding.", logger=preprocess_logger)
    except Exception as e:
        log_error(f"Error during OneHot Encoding: {e}", logger=preprocess_logger)
        raise e

    if TARGET_COL not in df.columns:
        log_error(f"FATAL: Target column '{TARGET_COL}' not found after preprocessing.", logger=preprocess_logger)
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    report["X"] = pd.DataFrame(X)
    report["y"] = pd.DataFrame(y)

    try:
        corr = df.corr(numeric_only=True)
        if TARGET_COL in corr.columns:
            target_corr = corr[TARGET_COL].drop(labels=[TARGET_COL], errors="ignore")
            low_corr = target_corr[abs(target_corr) < 0.02].index.tolist()
            if low_corr:
                X.drop(columns=low_corr, inplace=True, errors="ignore")
                log_info(f"Dropped low-correlation columns: {low_corr}", logger=preprocess_logger)
    except Exception as e:
        log_error(f"Error dropping low-correlation columns: {e}", logger=preprocess_logger)

    try:
        corr_limit = 0.75
        corr_filtered = X.corr().abs()
        upper = corr_filtered.where(np.triu(np.ones(corr_filtered.shape), k=1).astype(bool))
        drop_high = set()
        for col in upper.columns:
            high_pairs = upper[col][upper[col] > corr_limit]
            for idx, _ in high_pairs.items():
                corr_idx_target = abs(corr.loc[idx, TARGET_COL]) if idx in corr.index and TARGET_COL in corr.columns else 0
                corr_col_target = abs(corr.loc[col, TARGET_COL]) if col in corr.index and TARGET_COL in corr.columns else 0
                drop_col = idx if corr_idx_target < corr_col_target else col
                drop_high.add(drop_col)
        if drop_high:
            X.drop(columns=list(drop_high), inplace=True, errors="ignore")
            log_info(f"Dropped multicollinear columns: {sorted(list(drop_high))}", logger=preprocess_logger)
    except Exception as e:
        log_error(f"Error removing multicollinearity: {e}", logger=preprocess_logger)
        raise e

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if len(y.unique()) > 1 else None
        )
        log_info("Data split into train/test sets.", logger=preprocess_logger)
    except Exception as e:
        log_error(f"Error during train/test split: {e}", logger=preprocess_logger)
        raise e

    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        safe_save_joblib(scaler, MODELS_DIR / "scaler.joblib")
        log_info("StandardScaler applied and saved.", logger=preprocess_logger)
    except Exception as e:
        log_error(f"Error during scaling: {e}", logger=preprocess_logger)
        raise e

    try:
        split_data = (X_train_scaled, X_test_scaled, y_train.reset_index(drop=True), y_test.reset_index(drop=True))
        safe_save_joblib(split_data, MODELS_DIR / "data_splits.joblib")
        log_info(f"Processed data splits saved successfully (joblib). Train shape: {X_train_scaled.shape}", logger=preprocess_logger)

        if save_json:
            json_report = {
                "X_columns": X.columns.tolist(),
                "X_train_shape": list(X_train.shape),
                "X_test_shape": list(X_test.shape),
            }
            json_path = MODELS_DIR / "preprocessed_data_summary.json"
            save_to_json(json_report, json_path)
            log_info(f"JSON summary saved successfully at {json_path}", logger=preprocess_logger)
    except Exception as e:
        log_error(f"Error saving data splits: {e}", logger=preprocess_logger)
        raise e

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess diabetes dataset for ANN training.")
    parser.add_argument("--data_path", type=str, default=str(DATA_PATH), help="Path to CSV dataset file.")
    parser.add_argument("--save_json", action="store_true", help="Save preprocessing summary as JSON.")
    args = parser.parse_args()

    log_info("Running preprocess.py as standalone script.", logger=preprocess_logger)
    preprocess_split(data_path=Path(args.data_path), save_json=args.save_json)