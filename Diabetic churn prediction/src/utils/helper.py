from pathlib import Path
import joblib
import json
import numpy as np
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
    
from src.utils.logger import log_info,log_error,log_warning,helper_logger

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj,np.integer):
            return int(obj)
        if isinstance(obj,np.floating):
            return float(obj)
        if isinstance(obj,np.ndarray):
            return obj.tolist()
        return super(NpEncoder,self).default(obj)

def safe_save_joblib(obj, path: Path, compress=True):
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
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, cls=NpEncoder, indent=indent)
        log_info(f"JSON report saved successfully to: {path}",logger=helper_logger)
    except Exception as e:
        log_error(f"Failed to save JSON to {path}: {e}",logger=helper_logger)
        raise e

def find_latest_file(directory: Path, pattern: str) -> Path | None:
    """
    Finds the most recently modified file in a directory matching a pattern.
    """
    try:
        files = list(directory.glob(pattern))
        if not files:
            log_warning(f"No files found matching pattern '{pattern}' in {directory}", logger=helper_logger)
            return None
        
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        log_info(f"Found latest file: {latest_file}", logger=helper_logger)
        return latest_file
    except Exception as e:
        log_error(f"Error finding latest file with pattern '{pattern}': {e}", logger=helper_logger)
        return None