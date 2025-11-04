"""
Handles loading, cleaning, encoding, feature selection, and scaling of the churn dataset
based on the logic from the analysis notebook.

Saves encoder and scaler artifacts to models/ folder for Streamlit app usage.
"""
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from src.utils.logger import log_info, log_error, log_warning, preprocess_logger
from src.utils.helper import safe_save_joblib,save_to_json
import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE / "data" / "customer_churn_dataset-testing-master.csv"
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


report={}

DROP_COLS_PRE = ['CustomerID']
LABEL_ENCODE_COLS = ['Gender']
ONEHOT_ENCODE_COLS = ['Subscription Type', 'Contract Length']
TARGET_COL = 'Churn'

COLS_TO_DROP_POST_ENCODING = [
    'Age', 
    'Last Interaction',
    'Subscription Type_Basic', 
    'Subscription Type_Premium', 
    'Subscription Type_Standard',
    'Contract Length_Annual', 
    'Contract Length_Monthly', 
    'Contract Length_Quarterly'
]


COLS_TO_SCALE = [
    'Gender', 
    'Tenure', 
    'Usage Frequency', 
    'Support Calls', 
    'Payment Delay',
    'Total Spend'
]


def preprocess_and_split(save_json : bool=True):
    """
    Applies the full preprocessing pipeline from the notebook:
    1. Load Data
    2. Clean (Drop CustomerID, Fill NaNs)
    3. Encode (LabelEncoder, OneHotEncoder) on full dataset
    4. Save Encoders
    5. Feature Selection (Drop low-correlation columns)
    6. Split into Train/Test
    7. Scale (StandardScaler) on Train/Test splits
    8. Save Scaler and processed data splits
    """
    MODELS_DIR.mkdir(exist_ok=True)

    try:
        df = pd.read_csv(DATA_PATH)
        log_info(f"Dataset loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns", logger=preprocess_logger)
    except FileNotFoundError:
        log_error(f"FATAL: Dataset not found at {DATA_PATH}. Please add it to the 'data/' folder.", logger=preprocess_logger)
        raise
    except Exception as e:
        log_error(f"Failed to load dataset: {e}", logger=preprocess_logger)
        raise e

    try:
        if DROP_COLS_PRE[0] in df.columns:
            df.drop(columns=DROP_COLS_PRE, inplace=True)
            log_info(f"'{DROP_COLS_PRE[0]}' column dropped.", logger=preprocess_logger)
        
        if df.isnull().sum().any():
            log_warning("NaN values detected. Filling with median (numeric) or mode (categorical).", logger=preprocess_logger)
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
    except Exception as e:
        log_error(f"Error during basic cleaning: {e}", logger=preprocess_logger)
        raise e

    try:
        le = LabelEncoder()
        df[LABEL_ENCODE_COLS[0]] = le.fit_transform(df[LABEL_ENCODE_COLS[0]])
        safe_save_joblib(le, MODELS_DIR / "label_encoder.joblib")
        log_info("LabelEncoder applied to 'Gender' and saved.", logger=preprocess_logger)
    except Exception as e:
        log_error(f"Error during Label Encoding: {e}", logger=preprocess_logger)
        raise e

    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_array = encoder.fit_transform(df[ONEHOT_ENCODE_COLS])
        encoded_cols = encoder.get_feature_names_out(ONEHOT_ENCODE_COLS)
        
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
        
        df = pd.concat([df.drop(columns=ONEHOT_ENCODE_COLS), encoded_df], axis=1)
        safe_save_joblib(encoder, MODELS_DIR / "onehot_encoder.joblib")
        log_info("OneHotEncoder applied and saved.", logger=preprocess_logger)
        log_info(f"Columns after encoding: {df.columns.tolist()}", logger=preprocess_logger)
    except Exception as e:
        log_error(f"Error during OneHot Encoding: {e}", logger=preprocess_logger)
        raise e

    if TARGET_COL not in df.columns:
        log_error(f"FATAL: Target column '{TARGET_COL}' not found after preprocessing.", logger=preprocess_logger)
        raise ValueError(f"Target column '{TARGET_COL}' not found.")
        
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    report['X']=pd.DataFrame(X)
    report['y']=pd.DataFrame(y)
    try:
        cols_to_drop_final = [col for col in COLS_TO_DROP_POST_ENCODING if col in X.columns]
        X = X.drop(columns=cols_to_drop_final)
        log_info(f"Applied feature selection. Dropped: {cols_to_drop_final}", logger=preprocess_logger)
        log_info(f"Final features for training: {X.columns.tolist()}", logger=preprocess_logger)
    except Exception as e:
        log_error(f"Error during feature selection: {e}", logger=preprocess_logger)
        raise e
        
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        report['X_train']=pd.DataFrame(X_train)
        report['y_train']=pd.DataFrame(y_train)
        report['X_test']=pd.DataFrame(X_test)
        report['y_test']=pd.DataFrame(y_test)
        log_info("Data split into train/test sets.", logger=preprocess_logger)
    except Exception as e:
        log_error(f"Error during train/test split: {e}", logger=preprocess_logger)
        raise e
    

    cols_to_scale_final = [col for col in COLS_TO_SCALE if col in X_train.columns]
    
    try:
        scaler = StandardScaler()
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[cols_to_scale_final] = scaler.fit_transform(X_train[cols_to_scale_final])
        X_test_scaled[cols_to_scale_final] = scaler.transform(X_test[cols_to_scale_final])
        report['X_train_scaled']=X_train_scaled
        report['X_test_scaled'] = X_test_scaled
        
        safe_save_joblib(scaler, MODELS_DIR / "scaler.joblib")
        log_info("StandardScaler applied and saved.", logger=preprocess_logger)
    except Exception as e:
        log_error(f"Error during scaling: {e}", logger=preprocess_logger)
        raise e

    try:
        split_data = (X_train_scaled, X_test_scaled, y_train, y_test)
        safe_save_joblib(split_data, MODELS_DIR / "data_splits.joblib")
        log_info(f"Feature processing complete. Final train features shape: {X_train_scaled.shape}", logger=preprocess_logger)
        log_info("Processed data splits saved successfully (joblib).", logger=preprocess_logger)

        if save_json:
            try:
                json_report = {
                    "X": X.to_dict(orient="records"),
                    "y": y.to_list(),
                    "X_train": X_train.to_dict(orient="records"),
                    "y_train": y_train.to_list(),
                    "X_test": X_test.to_dict(orient="records"),
                    "y_test": y_test.to_list(),
                    "X_train_scaled": X_train_scaled.to_dict(orient="records"),
                    "X_test_scaled": X_test_scaled.to_dict(orient="records")
                }

                json_path = MODELS_DIR / "preprocessed_data.json"
                log_info(f"Attempting to save JSON at: {json_path}", logger=preprocess_logger)
                save_to_json(json_report, json_path)
                log_info(f"✅ JSON saved successfully at {json_path}", logger=preprocess_logger)
            except Exception as e:
                log_error(f"❌ Failed to save JSON: {e}", logger=preprocess_logger)
                raise e


    except Exception as e:
        log_error(f"Error saving data splits: {e}", logger=preprocess_logger)
        raise e

if __name__ == "__main__":
    log_info("Running preprocess.py as standalone script.", logger=preprocess_logger)
    preprocess_and_split(save_json=True)
