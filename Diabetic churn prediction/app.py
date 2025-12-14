import sys
from pathlib import Path
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import warnings

# === 1. FLASK IMPORTS ===
# We must import the functions for serving a webpage (render_template)
# and handling data (Flask, request, jsonify)
from flask import Flask, request, jsonify, render_template

# === 2. CORRECT PATH SETUP (THE MAIN FIX) ===
# This tells your `app.py` (in the root) how to find the `src` folder
# so it can import your helper and logger files.
BASE_DIR = Path(__file__).resolve().parent  # This is D:\diabeties_churmn_prediction
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Imports *must* come AFTER the path is fixed
from src.utils.logger import log_info, log_error, app_logger
from src.utils.helper import safe_load_joblib, find_latest_file

# === 3. SUPPRESS WARNINGS ===
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
tf.get_logger().setLevel('ERROR')

# === 4. LOAD ARTIFACTS ===
log_info("Initializing Flask app and loading artifacts...", logger=app_logger)
MODELS_DIR = BASE_DIR / "models"
MODEL_PATTERN = "ann_model_*.keras"  # Using the .keras fix

try:
    # Load Model
    latest_model_path = find_latest_file(MODELS_DIR, MODEL_PATTERN)
    if not latest_model_path:
        log_error(f"FATAL: No model file matching '{MODEL_PATTERN}' found in {MODELS_DIR}", logger=app_logger)
        log_error("Please run 'python main.py' first to train and save a model.", logger=app_logger)
        sys.exit(1)
    MODEL = tf.keras.models.load_model(latest_model_path)
    log_info(f"Loaded model: {latest_model_path}", logger=app_logger)

    # Load Preprocessing Artifacts
    SCALER = safe_load_joblib(MODELS_DIR / "scaler.joblib")
    OHE = safe_load_joblib(MODELS_DIR / "onehot_encoder.joblib")
    CAT_COLS = safe_load_joblib(MODELS_DIR / "categorical_columns.joblib")
    NUM_COLS = safe_load_joblib(MODELS_DIR / "numerical_columns.joblib")
    
    summary_path = MODELS_DIR / "preprocessed_data_summary.json"
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
        FINAL_COLUMNS = summary_data["X_columns"]
    
    log_info("All artifacts (model, scaler, ohe, column lists) loaded successfully.", logger=app_logger)
    
except FileNotFoundError as e:
    log_error(f"FATAL: Missing a necessary file: {e}", logger=app_logger)
    log_error("Please run 'python main.py' to generate all artifacts.", logger=app_logger)
    sys.exit(1)
except Exception as e:
    log_error(f"FATAL: Failed to load artifacts on startup: {e}", logger=app_logger)
    sys.exit(1)


# === 5. INITIALIZE FLASK APP ===
# This tells Flask where to find 'static' and 'templates' folders
app = Flask(__name__, static_folder='static', template_folder='templates')


# === 6. PREPROCESSING FUNCTION (Unchanged) ===
# This function prepares the data from the webpage for the model
def preprocess_input(data: dict) -> np.ndarray:
    try:
        input_df = pd.DataFrame([data])
        # Handle Missing Values (replicating preprocess.py logic)
        for col in input_df.columns:
            if col in CAT_COLS:
                mode_val = input_df[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else ""
                input_df[col].fillna(fill_val, inplace=True)
            elif col in NUM_COLS:
                fill_val = input_df[col].median()
                if pd.isna(fill_val): # Handle case where median is also NaN
                     fill_val = 0
                input_df[col].fillna(fill_val, inplace=True)
        
        input_df.fillna(0, inplace=True) # Fallback for any other NaNs
        
        # Apply One-Hot Encoding
        input_cat_df = pd.DataFrame()
        if CAT_COLS:
            input_cat = input_df[CAT_COLS]
            input_cat_processed = OHE.transform(input_cat)
            input_cat_df = pd.DataFrame(input_cat_processed, columns=OHE.get_feature_names_out(CAT_COLS))

        # Combine processed categorical with all other columns
        input_df_processed = pd.concat([input_df.drop(columns=CAT_COLS, errors='ignore'), input_cat_df], axis=1)

        # Reorder and Select FINAL_COLUMNS
        final_df_template = pd.DataFrame(columns=FINAL_COLUMNS)
        # Use concat instead of reindex to handle new/missing columns gracefully
        input_reordered = pd.concat([final_df_template, input_df_processed], ignore_index=True)
        input_to_scale = input_reordered[FINAL_COLUMNS].head(1).fillna(0)
        
        # Apply Scaling
        input_scaled = SCALER.transform(input_to_scale)
        return input_scaled
    except Exception as e:
        log_error(f"Error in preprocess_input: {e}", logger=app_logger)
        raise e

# === 7. API ROUTES (THE WEB SERVER LOGIC) ===

@app.route("/")
def index():
    """
    This is the main webpage route (the fix for your error).
    When a user visits http://127.0.0.1:5000/, this function runs
    and serves the 'index.html' file from the 'templates' folder.
    """
    try:
        log_info("Serving index.html", logger=app_logger)
        return render_template("index.html")
    except Exception as e:
        log_error(f"Error rendering index.html: {e}", logger=app_logger)
        return "Error: Could not find template 'index.html'. Please check your 'templates' folder.", 500

@app.route("/predict", methods=["POST"])
def predict():
    """
    This route handles the prediction. The JavaScript from index.html
    sends its form data here.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    log_info(f"Received prediction request: {data}", logger=app_logger)

    try:
        # Check for *at least* the required keys
        required_keys = set(CAT_COLS + NUM_COLS)
        if not required_keys.issubset(data.keys()):
            missing = required_keys - set(data.keys())
            log_error(f"Bad request. Missing keys: {missing}", logger=app_logger)
            return jsonify({"error": "Bad request", "missing_keys": list(missing)}), 400

        # Process the input
        processed_data = preprocess_input(data)

        # Make prediction
        prediction_prob = MODEL.predict(processed_data)[0][0]
        prediction_class = 1 if prediction_prob > 0.5 else 0
        prediction_label = "Diabetes" if prediction_class == 1 else "No Diabetes"

        response = {
            "prediction": prediction_label,
            "prediction_class": prediction_class,
            "probability": float(prediction_prob)
        }
        
        log_info(f"Prediction successful: {response}", logger=app_logger)
        return jsonify(response), 200

    except Exception as e:
        log_error(f"Error during prediction: {e}", logger=app_logger)
        return jsonify({"error": "An error occurred during prediction."}), 500

# === 8. RUN THE APP ===
if __name__ == "__main__":
    # This runs the Flask web server
    app.run(debug=True, port=5000, host="0.0.0.0")