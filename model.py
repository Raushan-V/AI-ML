import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# 1. Data Collection
def load_data(filepath='../creditcard.csv'):
    df = pd.read_csv(filepath)
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    return df

# 2. Exploratory Data Analysis (EDA)
def perform_eda(df):
    print("Class distribution:")
    print(df['Class'].value_counts())
    print("Percentage of fraud:", df['Class'].value_counts()[1] / len(df) * 100)

    # Visualize class distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution')
    plt.savefig('class_distribution.png')
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12,10))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.show()

    # Time and Amount distributions
    fig, ax = plt.subplots(1, 2, figsize=(14,4))
    sns.histplot(df['Time'], ax=ax[0], kde=True)
    ax[0].set_title('Time Distribution')
    sns.histplot(df['Amount'], ax=ax[1], kde=True)
    ax[1].set_title('Amount Distribution')
    plt.savefig('time_amount_dist.png')
    plt.show()

# 3. Handle Imbalanced Data
def handle_imbalance(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print("After SMOTE - X shape:", X_res.shape, "y shape:", y_res.shape)
    print("Resampled class distribution:", pd.Series(y_res).value_counts())
    return X_res, y_res

# 4. Model Development
def train_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'LightGBM': LGBMClassifier(random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:,1]
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'report': classification_report(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred)
        }
        print(f"{name} - ROC-AUC: {results[name]['roc_auc']:.4f}, F1: {results[name]['f1']:.4f}")
    return results

# Hyperparameter tuning with Optuna
def tune_xgboost(X_train, y_train):
    def objective(trial):
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0)
        }
        model = XGBClassifier(**param, random_state=42)
        model.fit(X_train, y_train)
        return model.score(X_train, y_train)  # Use validation score in practice

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    return study.best_params

# 5. Anomaly Detection Models
def train_anomaly_models(X_train, X_test):
    anomaly_models = {
        'Isolation Forest': IsolationForest(random_state=42),
        'One-Class SVM': OneClassSVM(kernel='rbf', gamma='auto')
    }

    anomaly_results = {}
    for name, model in anomaly_models.items():
        model.fit(X_train)
        y_pred = model.predict(X_test)
        # Convert to binary: -1 for anomaly (fraud), 1 for normal
        y_pred_binary = np.where(y_pred == -1, 1, 0)
        anomaly_results[name] = y_pred_binary
    return anomaly_results

# Autoencoder for anomaly detection
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    autoencoder = build_autoencoder(X_train.shape[1])
    autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=256, validation_split=0.1, verbose=0)

    # Reconstruction error
    X_test_pred = autoencoder.predict(X_test_scaled)
    mse = np.mean(np.power(X_test_scaled - X_test_pred, 2), axis=1)
    threshold = np.percentile(mse, 95)  # 95th percentile as threshold
    y_pred = (mse > threshold).astype(int)
    return y_pred

# 6. Model Evaluation
def evaluate_models(results, y_test, anomaly_results=None):
    for name, res in results.items():
        print(f"\n{name} Classification Report:")
        print(res['report'])

        # Confusion Matrix
        cm = confusion_matrix(y_test, res['y_pred'])
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.savefig(f'{name.lower().replace(" ", "_")}_cm.png')
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f'{name} (AUC = {res["roc_auc"]:.2f})')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} ROC Curve')
        plt.legend()
        plt.savefig(f'{name.lower().replace(" ", "_")}_roc.png')
        plt.show()

    if anomaly_results:
        for name, y_pred in anomaly_results.items():
            print(f"\n{name} Anomaly Detection Report:")
            print(classification_report(y_test, y_pred))

# 7. Model Monitoring with MLflow
def log_with_mlflow(model, name, params, metrics):
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name)

# Main function
if __name__ == "__main__":
    # Load data
    df = load_data()

    # EDA
    perform_eda(df)

    # Prepare data
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle imbalance
    X_train_res, y_train_res = handle_imbalance(X_train_scaled, y_train)

    # Train classification models
    results = train_models(X_train_res, y_train_res, X_test_scaled, y_test)

    # Tune XGBoost
    best_params = tune_xgboost(X_train_res, y_train_res)
    tuned_xgb = XGBClassifier(**best_params, random_state=42)
    tuned_xgb.fit(X_train_res, y_train_res)
    print("Best XGBoost params:", best_params)

    # Anomaly detection
    anomaly_results = train_anomaly_models(X_train_scaled, X_test_scaled)
    autoencoder_pred = train_autoencoder(X_train_scaled, X_test_scaled)
    anomaly_results['Autoencoder'] = autoencoder_pred

    # Evaluate
    evaluate_models(results, y_test, anomaly_results)

    # Log with MLflow (example for one model)
    log_with_mlflow(results['XGBoost']['model'], 'xgboost_model', {}, {'roc_auc': results['XGBoost']['roc_auc']})

    print("Fraud detection model training and evaluation completed.")
