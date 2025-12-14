import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.utils.logger import app_logger, log_info, log_error, log_warning

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Customer Churn Prediction",
                   page_icon="📊",
                   layout="centered")

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("""
Welcome to the Customer Churn Prediction System.  
Enter customer details below to predict whether a customer is likely to **churn** or **stay**.
""")
st.divider()

st.sidebar.header("📂 About the Project")
st.sidebar.markdown("""
This app uses a **Machine Learning model (Random Forest / sklearn)**  
to predict whether a customer will **churn** based on behavior and demographics.
""")
st.sidebar.info("Developed by **Om Babhulkar**")

REQUIRED_FILES = {
    "model": "churn_model.joblib",
    "label_encoder": "label_encoder.joblib",
    "scaler": "scaler.joblib",
    "onehot_encoder": "onehot_encoder.joblib"
}

LABEL_ENCODE_COLS = ['Gender']
ONEHOT_ENCODE_COLS = ['Subscription Type', 'Contract Length']
NUMERIC_COLS = ['Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend']
COLS_TO_USE = ['Gender'] + NUMERIC_COLS

@st.cache_resource
def load_artifact(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {file_path.name}")
    return joblib.load(file_path)

def load_all_artifacts():
    artifacts = {}
    missing_files = []
    for key, filename in REQUIRED_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            missing_files.append(filename)
            log_error(f"Missing artifact: {filename}", app_logger)
        else:
            artifacts[key] = load_artifact(path)
            log_info(f"Loaded artifact: {filename}", app_logger)
    if missing_files:
        st.error(f"Missing files: {', '.join(missing_files)}. Please train the model again.")
        return None
    log_info("All model artifacts loaded successfully.", app_logger)
    return artifacts

artifacts = load_all_artifacts()

def predict_churn(input_data: pd.DataFrame, artifacts: dict):
    try:
        df = input_data.copy()
        le = artifacts['label_encoder']
        df['Gender'] = le.transform(df['Gender'])
        log_info(f"Gender encoded: {df['Gender'].tolist()}", app_logger)

        X = df[['Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend',
                'Subscription Type', 'Contract Length']]

        scaler = artifacts['scaler']
        scaled_part = scaler.transform(
            X[['Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend']]
        )

        ohe = artifacts['onehot_encoder']
        ohe_part = ohe.transform(X[ONEHOT_ENCODE_COLS])

        final_input = np.hstack((scaled_part))

        if final_input.ndim == 1:
            final_input = final_input.reshape(1, -1)

        model = artifacts['model']
        proba = model.predict_proba(final_input)[0]

        prob_churn = proba[list(model.classes_).index(1)]

        threshold = 0.45
        prediction = 1 if prob_churn >= threshold else 0

        log_info(
            f"Prediction using threshold {threshold} | Prob(Churn)={prob_churn:.4f} | Predicted={prediction}",
            app_logger
        )

        if 0.45 <= prob_churn <= 0.55:
            st.warning(f"⚠️ Borderline case — model uncertain (Churn probability: {prob_churn:.2%})")

        return prediction, proba, final_input

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        log_error(f"Error during preprocessing or prediction: {e}", app_logger)
        return None, None, None


if artifacts:
    with st.form("churn_prediction_form"):
        st.subheader("Customer Details")

        col1, col2, col3 = st.columns(3)
        with col1:
            Age = st.number_input("Age", min_value=18, max_value=100, value=30)
            Gender = st.selectbox("Gender", ["Male", "Female"])
            Tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
            Usage_Frequency = st.number_input("Usage Frequency (per month)", min_value=0, value=15)

        with col2:
            Support_Calls = st.number_input("Support Calls", min_value=0, value=2)
            Payment_Delay = st.number_input("Payment Delay (days)", min_value=0, value=0)
            Subscription_Type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])

        with col3:
            Contract_Length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
            Total_Spend = st.number_input("Total Spend ($)", min_value=0.0, value=100.0, step=10.0)
            Last_Interaction = st.number_input("Last Interaction (days ago)", min_value=0, value=10)

        st.divider()
        submit = st.form_submit_button("Predict Churn", use_container_width=True)

    if submit:
        input_data = pd.DataFrame([{
            'Age': Age,
            'Gender': Gender,
            'Tenure': Tenure,
            'Usage Frequency': Usage_Frequency,
            'Support Calls': Support_Calls,
            'Payment Delay': Payment_Delay,
            'Subscription Type': Subscription_Type,
            'Contract Length': Contract_Length,
            'Total Spend': Total_Spend,
            'Last Interaction': Last_Interaction
        }])

        log_info(f"Received input: {input_data.to_dict(orient='records')}", app_logger)

        prediction, proba, processed = predict_churn(input_data, artifacts)

        if prediction is not None:
            churn_prob = proba[list(artifacts['model'].classes_).index(1)]
            if prediction == 1:
                st.error(f"🚨 The customer is **likely to CHURN** (Probability: {churn_prob:.2%})")
                log_warning(f"Customer likely to churn ({churn_prob:.2%})", app_logger)
            else:
                st.success(f"✅ The customer is **not likely to churn** (Probability: {churn_prob:.2%})")
                log_info(f"Customer not likely to churn ({churn_prob:.2%})", app_logger)

            with st.expander("🔧 Show processed data (debug view)"):
                st.dataframe(pd.DataFrame(processed))
        else:
            log_warning("Prediction failed or returned None", app_logger)
else:
    st.info("Model artifacts not found. Please train the model first.")
    log_warning("Model artifacts not found. Application stopped.", app_logger)
