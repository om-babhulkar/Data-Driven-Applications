# 🧠 Customer Churn Prediction System
A complete **end-to-end Machine Learning web application** that predicts whether a customer will churn (leave the service) based on behavioral and demographic data.

---

## 🚀 **Project Overview**

Customer churn is one of the biggest challenges for subscription-based businesses.  
This project demonstrates how to:
- Build and preprocess a customer dataset  
- Train and evaluate a classification model (Random Forest / Scikit-learn)  
- Deploy the model via a **Streamlit web dashboard**  
- Log activities and maintain modular pipeline structure  
- Write **unit tests (pytest)** for reliability  

---

## 🧩 **Tech Stack**

| Area | Technology Used |
|-------|-----------------|
| Language | Python 3.10+ |
| Framework | Streamlit |
| Machine Learning | Scikit-learn, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Persistence | Joblib |
| Logging | Python `logging` module |
| Testing | Pytest |
| Environment | `.env`, Virtualenv |
| Version Control | Git & GitHub |

---

## 📂 **Project Structure**

customer_churn_prediction/
│
├── src/
│ ├── ml_pipeline/
│ │ ├── preprocess.py # Data preprocessing and feature engineering
│ │ ├── train_model.py # Model training and saving
│ │ ├── test.py # Model evaluation and metrics report
│ │ └── init.py
│ │
│ └── utils/
│ ├── helper.py # Helper functions (save/load artifacts)
│ ├── logger.py # Centralized logging configuration
│ └── init.py
│
├── models/ # Contains trained models & encoders
│ ├── churn_model.joblib
│ ├── scaler.joblib
│ ├── label_encoder.joblib
│ ├── onehot_encoder.joblib
│ └── evaluation_report.json
│
├── tests/ # Unit tests for each module
│ ├── test_artifacts.py
│ ├── test_preprocess.py
│ └── test_ui.py
│
├── app.py # Streamlit web application
├── main.py # Entry script to run pipeline
├── requirements.txt # Project dependencies
├── .gitignore # Ignore unwanted files/folders
└── README.md # Documentation

---

## 🧠 **Model Pipeline**

1. **Preprocessing**
   - Cleans dataset, handles categorical encoding (LabelEncoder, OneHotEncoder)
   - Scales numeric features with StandardScaler  
   - Splits data into train/test sets  

2. **Training**
   - Trains a Random Forest classifier  
   - Saves model & transformers as `.joblib` artifacts  

3. **Evaluation**
   - Calculates accuracy, F1, precision, recall, and ROC-AUC  
   - Saves results to `evaluation_report.json`  

4. **Deployment**
   - Interactive Streamlit app for user input  
   - Predicts churn probability in real time  

📊 Evaluation Metrics
Metric	Score (Example)
Accuracy	    93.4%
Precision	    94.9%
Recall	        92.3%
F1-Score	    93.6%
ROC-AUC	        0.94


📦 Future Improvements

1.Add SHAP or LIME for explainable AI insights

2.Integrate CI/CD pipeline with GitHub Actions

3.Add Docker support for deployment

4.Enhance dataset handling via database integration (PostgreSQL/Firebase)

5.Cloud deployment using Streamlit Cloud or AWS EC2


👨‍💻 Author

Om Babhulkar
📍 Maharashtra, India
🎓 B.Tech in Information Technology, GCOE Amravati
💡 Aspiring AI/ML Engineer & Full Stack python Developer
