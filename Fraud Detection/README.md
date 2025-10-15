Credit Card Fraud Detection

This project implements a comprehensive fraud detection system using machine learning techniques on the Credit Card Fraud Detection dataset from Kaggle.

Project Structure

fraud detection/
├── creditcard.csv          # Dataset
├── fraud detection/
│   ├── model.py            # Main model training and evaluation script
│   ├── app.py              # FastAPI REST API for predictions
│   ├── dashboard.py        # Streamlit dashboard for visualization
│   ├── monitoring.py       # Model monitoring and drift detection
│   ├── requirements.txt    # Python dependencies
│   ├── TODO.md             # Project progress tracker
│   └── README.md           # This file

Workflow

1. Data Collection: Load and explore the credit card transaction dataset
2. EDA: Analyze class distribution, correlations, and patterns
3. Data Preprocessing: Handle imbalanced data with SMOTE, scale features
4. Model Development: Train multiple ML models (Logistic Regression, Random Forest, XGBoost, LightGBM)
5. Anomaly Detection: Implement Isolation Forest, One-Class SVM, and Autoencoder
6. Model Evaluation: Compare performance using ROC-AUC, F1-score, confusion matrices
7. Deployment: REST API with FastAPI and interactive dashboard with Streamlit
8. Monitoring: Real-time model monitoring with drift detection and alerting

Installation

1. Install dependencies:
pip install -r requirements.txt

2. Run the main model:
python model.py

Usage

Training Models
Run the main script to train and evaluate models:
python model.py

API Server
Start the FastAPI server:
uvicorn app:app --reload
The API will be available at http://localhost:8000

Dashboard
Launch the Streamlit dashboard:
streamlit run dashboard.py

Models Implemented

Classification Models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM

Anomaly Detection:
  - Isolation Forest
  - One-Class SVM
  - Autoencoder (Neural Network)

Evaluation Metrics

- ROC-AUC Score
- F1-Score
- Precision-Recall Curve
- Confusion Matrix

Results

The models achieve high ROC-AUC scores (>0.97) on the test set. Due to the imbalanced nature of the dataset, F1-score is used as the primary metric for fraud detection performance.

Technologies Used

- Python 3.8+
- scikit-learn
- XGBoost, LightGBM
- TensorFlow/Keras
- FastAPI
- Streamlit
- pandas, numpy, matplotlib, seaborn

Dataset

The dataset contains transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.

Features:
- Time: Number of seconds elapsed between this transaction and the first transaction
- V1-V28: Principal components obtained with PCA
- Amount: Transaction amount
- Class: 1 for fraudulent, 0 otherwise

Model Monitoring

The system includes comprehensive monitoring capabilities:

Prediction Drift Detection: Monitors changes in model performance over time
Data Drift Detection: Detects shifts in input data distribution
Performance Logging: Tracks F1-score and ROC-AUC metrics
Alerting System: Automatic alerts for unusual prediction patterns
Monitoring Dashboard: Integrated monitoring reports in the Streamlit dashboard

Future Improvements

- Implement advanced monitoring with Evidently AI
- Add more anomaly detection techniques
- Deploy to cloud platform
- Real-time prediction pipeline
- A/B testing framework

Q&A

Q: How was this project created?
A: This project was developed step-by-step starting with data collection from the Kaggle Credit Card Fraud Detection dataset. We performed exploratory data analysis to understand the data distribution and correlations. Then, we handled the imbalanced dataset using SMOTE, trained multiple machine learning models including classification and anomaly detection algorithms, evaluated their performance, and finally deployed the system with a REST API and interactive dashboard. Model monitoring was added to track performance drift and data drift in production.

Q: What are the pros and cons of the dataset?
A: Pros: The dataset represents real-world credit card transactions with anonymized features to protect privacy, making it suitable for fraud detection research. It has a large number of transactions (284,807) and includes both time and amount features which are important for fraud analysis. Cons: The dataset is highly imbalanced with only 0.172% fraud cases, which can lead to biased models if not handled properly. The anonymized features (V1-V28) limit interpretability since we don't know what they represent. Additionally, the dataset is from 2013, so it may not reflect current fraud patterns.

License

This project is for educational purposes.
