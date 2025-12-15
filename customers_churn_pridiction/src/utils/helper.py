"""
Utility functions for safely saving and loading
ML artifacts using joblib.
"""
from pathlib import Path
import joblib
import json
import numpy as np
from .logger import log_info, log_error,helper_logger

class NpEncoder(json.JSONEncoder):
    """ Custom JSON encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def safe_save_joblib(obj, path: Path, compress=True):
    """
    Safely saves a large ML object (like trained models) using joblib.
    Optionally compresses for smaller file size.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if compress:
            joblib.dump(obj, path, compress=("zlib", 3))
        else:
            joblib.dump(obj, path)
        log_info(f"Artifact saved successfully to: {path}",logger=helper_logger)
    except Exception as e:
        log_error(f"Failed to save joblib file at {path}: {e}",logger=helper_logger)
        raise e

def safe_load_joblib(path: Path):
    """
    Safely loads a joblib file.
    """
    if not path.exists():
        log_error(f"Joblib file not found: {path}",logger=helper_logger)
        raise FileNotFoundError(f"Joblib file not found: {path}")
    
    try:
        obj = joblib.load(path)
        log_info(f"Artifact loaded successfully from: {path}",logger=helper_logger)
        return obj
    except Exception as e:
        log_error(f"Failed to load joblib file from {path}: {e}",logger=helper_logger)
        raise e

def save_to_json(data, path: Path, indent: int = 4):
    """
    Saves data (like metrics) to a JSON file.
    Uses custom encoder for numpy types.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, cls=NpEncoder, indent=indent)
        log_info(f"JSON report saved successfully to: {path}",logger=helper_logger)
    except Exception as e:
        log_error(f"Failed to save JSON to {path}: {e}",logger=helper_logger)
        raise e
