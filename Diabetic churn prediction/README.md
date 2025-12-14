End-to-End Diabetes Prediction (ANN)

This is a complete, end-to-end data science project that predicts the likelihood of diabetes based on a range of medical and lifestyle data. It features a full MLOps pipeline, from data preprocessing and feature engineering to model training, evaluation, and deployment as an interactive web application.

The final model, a trained Artificial Neural Network (ANN), achieved 91.6% accuracy and a ROC-AUC of 0.942 on the test set.

The project is built to be data-driven. The preprocessing pipeline (preprocess.py) automatically discovers and removes features with low correlation or high multicollinearity, and the prediction pipeline (app.py) automatically adapts to these decisions.

Features

End-to-End Pipeline: A single command (python main.py) runs the entire data preprocessing, model training, and evaluation pipeline.

Data-Driven Feature Selection: The preprocessing script automatically identifies and drops features (including one-hot encoded ones) that are determined to be "noise" (low correlation or multicollinear).

Interactive Web Application: A Flask server (app.py) serves a user-friendly HTML/CSS interface (index.html) for live, real-time predictions.

High-Performance Model: Uses a TensorFlow/Keras ANN, achieving 91.6% accuracy.

Comprehensive Logging: Every script (main, app, preprocess, train, test) writes to its own log file in the /logs directory for easy debugging.

Full Test Suite: Includes a tests/ folder with pytest for smoke tests (checking imports) and API tests (checking the live Flask server).

D:\DIABETIES_CHURMN_PREDICTION
│
├── app.py                  
├── main.py                
├── requirements.txt        
├── .gitignore              
├── diabetes_dataset.csv   
│
├── src/
│   ├── ml_pipelines/
│   │   ├── preprocess.py     
│   │   ├── train.py          
│   │   └── test.py           
│   └── utils/
│       ├── helper.py         
│       └── logger.py         
│
├── templates/
│   └── index.html            
│
├── static/
│   └── style.css            
│
└── tests/
    ├── test_smoke.py        
    └── test_api.py           


How to Use

1. Installation

Clone the repository:

git clone 
cd your-project


Install the required libraries:

pip install -r requirements.txt


2. Running the Application (3 Steps)

Step 1: Run the Training Pipeline

First, you must run the main pipeline to generate all the artifacts (the model, scaler, etc.).

python main.py


This script will:

Read diabetes_dataset.csv.

Run the full preprocessing and feature selection.

Train the ANN model.

Save all artifacts (e.g., ann_model.keras, scaler.joblib, preprocessed_data_summary.json) into the /models folder.

Save all logs to the /logs folder.

Step 2: Run the Web Application

Once the pipeline is complete and the artifacts are saved, start the Flask web server.

python app.py


This will start the server, which loads the artifacts from /models and begins listening on http://127.0.0.1:5000.

Step 3: Use the Web Application

Open your web browser and navigate to:
http://127.0.0.1:5000

You will see the web form. You can fill it out (or just use the default test data) and click "Predict" to get a live prediction from your model.

3. Running the Test Suite (Optional)

This project uses pytest to ensure code quality.

In one terminal, start the web application:

python app.py


In a second terminal, run pytest:

pytest


This will run all tests in the tests/ folder. It will check that all scripts are importable (test_smoke.py) and that the live API is responding correctly to good and bad data (test_api.py).

Core Workflow: The "Smart-Filter" Pipeline

A key feature of this project is its robust, data-driven pipeline. The webpage asks for data (like gender) that the final model does not use. This is by design.

1. The "Training Funnel" (preprocess.py)

Input: Raw data (e.g., gender: "Male").

Step 1 (Transform): The OneHotEncoder is trained on the gender column. It learns to create new columns like gender_Male.

Step 2 (Judge): The script then calculates the correlation of all features. Your log (preprocess.log) confirms that gender_Male was judged to be "noise" (low correlation).

Step 3 (Filter): The script drops gender_Male for being useless.

Output: The script saves a "master recipe" (FINAL_COLUMNS in a .json file) that lists only the "good" features that passed the test. gender_Male is not on this list.

2. The "Prediction Funnel" (app.py)

Input: The user enters gender: "Male" on the webpage. This is required because the OneHotEncoder brain was trained to expect it.

Step 1 (Transform): app.py loads the saved OneHotEncoder and transforms gender: "Male" into gender_Male: 1.

Step 2 (Filter): app.py loads the FINAL_COLUMNS "master recipe". It filters the data, keeping only the columns on that list.

Output: gender_Male is silently and automatically dropped at this step because it's not on the list. Only the "good" data is sent to the model for prediction.

This "filter" system allows the preprocessing logic to be complex, but the final prediction pipeline remains fast, robust, and perfectly aligned with the data the model was trained on.

Technology Stack

Backend: Python, Flask

Data Science: Pandas, NumPy

Machine Learning: TensorFlow (Keras), scikit-learn

Frontend: HTML, CSS

Testing: pytest