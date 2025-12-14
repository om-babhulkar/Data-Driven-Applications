import pytest
import requests

# URL of the running Flask app
API_URL = "http://127.0.0.1:5000/predict"

@pytest.fixture(scope="module")
def api_server_is_running():
    """Checks if the API server is reachable before running tests."""
    try:
        response = requests.get("http://127.0.0.1:5000/")
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        pytest.fail(
            "ConnectionError: Could not connect to the API server at http://127.0.0.1:5000. "
            "Please ensure 'app.py' is running in a separate terminal before starting tests."
        )

# Test case for a 'No Diabetes' prediction
def test_predict_no_diabetes(api_server_is_running):
    patient_no_diabetes = {
        "age": 48, "gender": "Female", "ethnicity": "White", "smoking_status": "Former",
        "alcohol_consumption_per_week": 1, "physical_activity_minutes_per_week": 143,
        "diet_score": 6.7, "sleep_hours_per_day": 6.5, "screen_time_hours_per_day": 8.7,
        "family_history_diabetes": 0, "hypertension_history": 0, "cardiovascular_history": 0,
        "bmi": 23.1, "waist_to_hip_ratio": 0.8, "systolic_bp": 129, "diastolic_bp": 76,
        "heart_rate": 67, "cholesterol_total": 116, "hdl_cholesterol": 55, "ldl_cholesterol": 50,
        "triglycerides": 30, "glucose_fasting": 93, "glucose_postprandial": 150,
        "insulin_level": 2.0, "hba1c": 5.63, "diabetes_risk_score": 23.0
    }
    
    response = requests.post(API_URL, json=patient_no_diabetes)
    
    assert response.status_code == 200
    data = response.json()
    assert data['prediction_class'] == 0
    assert data['prediction'] == "No Diabetes"

# Test case for a 'Diabetes' prediction
def test_predict_diabetes(api_server_is_running):
    patient_diabetes = {
        "age": 58, "gender": "Male", "ethnicity": "Asian", "smoking_status": "Never",
        "alcohol_consumption_per_week": 0, "physical_activity_minutes_per_week": 215,
        "diet_score": 5.7, "sleep_hours_per_day": 7.9, "screen_time_hours_per_day": 7.9,
        "family_history_diabetes": 0, "hypertension_history": 0, "cardiovascular_history": 0,
        "bmi": 30.5, "waist_to_hip_ratio": 0.89, "systolic_bp": 134, "diastolic_bp": 78,
        "heart_rate": 68, "cholesterol_total": 239, "hdl_cholesterol": 41, "ldl_cholesterol": 160,
        "triglycerides": 145, "glucose_fasting": 136, "glucose_postprandial": 236,
        "insulin_level": 6.36, "hba1c": 8.18, "diabetes_risk_score": 29.6
    }
    
    response = requests.post(API_URL, json=patient_diabetes)
    
    assert response.status_code == 200
    data = response.json()
    assert data['prediction_class'] == 1
    assert data['prediction'] == "Diabetes"

# Test for bad request (missing keys)
def test_bad_request(api_server_is_running):
    bad_data = {"age": 50, "bmi": 30} # Missing all other keys
    
    response = requests.post(API_URL, json=bad_data)
    
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "missing_keys" in data