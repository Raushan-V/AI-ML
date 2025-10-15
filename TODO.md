# Fraud Detection Project TODO

## 1. Data Collection
- [x] Download the dataset (creditcard.csv available)
- [x] Explore features (time, amount, anonymized features, class labels)
- [x] Split into train/test sets

## 2. Exploratory Data Analysis (EDA)
- [x] Check class distribution (fraud vs non-fraud)
- [x] Visualize transaction patterns
- [x] Identify outliers and correlations
- [x] Normalize/scale numerical features

## 3. Handle Imbalanced Data
- [x] Apply SMOTE (Synthetic Minority Oversampling Technique)
- [ ] Try ADASYN, RandomUnderSampler, or SMOTEENN
- [x] Evaluate impact using ROC-AUC, F1-score

## 4. Model Development
- [x] Train Logistic Regression
- [x] Train Random Forest
- [x] Train XGBoost
- [x] Train LightGBM
- [x] Tune hyperparameters using GridSearchCV or Optuna

## 5. Anomaly Detection Models
- [x] Train Isolation Forest
- [x] Train One-Class SVM
- [x] Train Autoencoder (Deep Learning)
- [x] Compare detection accuracy against classification models

## 6. Model Evaluation
- [x] Compute Precision, Recall, F1-score
- [x] Plot ROC-AUC Curve
- [x] Plot Precision-Recall Curve
- [x] Generate Confusion Matrix

## 7. Model Monitoring (Post-Deployment)
- [x] Track prediction drift and data drift
- [x] Log model performance over time
- [x] Set alerts for unusual prediction patterns

## 8. Deployment (Optional Advanced Step)
- [x] Wrap the model in a REST API using FastAPI or Flask
- [x] Build a Streamlit dashboard to visualize fraud detection
- [x] Connect with monitoring tools for live updates

## 9. Documentation & Reporting
- [x] Create project README with workflow diagram
- [x] Save results and plots
- [x] Write a conclusion comparing different models
