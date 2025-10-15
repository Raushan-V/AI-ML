import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(filename='fraud_monitoring.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class FraudMonitor:
    def __init__(self, model_path='model.pkl', scaler_path='scaler.pkl'):
        self.model = None
        self.scaler = None
        self.performance_history = []
        self.prediction_log = []
        self.alert_threshold = 0.1  # Alert if performance drops by 10%

        # Load model and scaler if available
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        except FileNotFoundError:
            logging.warning("Model or scaler files not found. Monitoring will work with provided predictions.")

    def log_prediction(self, features, prediction, probability, actual=None):
        """Log a prediction for monitoring"""
        log_entry = {
            'timestamp': datetime.now(),
            'features': features.copy(),
            'prediction': prediction,
            'probability': probability,
            'actual': actual
        }
        self.prediction_log.append(log_entry)

        # Keep only last 10000 predictions for memory efficiency
        if len(self.prediction_log) > 10000:
            self.prediction_log = self.prediction_log[-10000:]

        logging.info(f"Prediction logged: {prediction}, Probability: {probability:.4f}")

    def check_performance_drift(self, test_data=None, test_labels=None):
        """Check for performance drift using recent predictions"""
        if len(self.prediction_log) < 100:
            return "Insufficient data for drift detection"

        recent_predictions = self.prediction_log[-1000:]  # Last 1000 predictions
        recent_actuals = [p['actual'] for p in recent_predictions if p['actual'] is not None]
        recent_preds = [p['prediction'] for p in recent_predictions if p['actual'] is not None]

        if len(recent_actuals) < 50:
            return "Insufficient labeled data for drift detection"

        current_f1 = f1_score(recent_actuals, recent_preds)
        current_roc_auc = roc_auc_score(recent_actuals, recent_preds)

        # Compare with baseline (first 100 predictions)
        baseline_predictions = self.prediction_log[:100]
        baseline_actuals = [p['actual'] for p in baseline_predictions if p['actual'] is not None]
        baseline_preds = [p['prediction'] for p in baseline_predictions if p['actual'] is not None]

        if len(baseline_actuals) >= 50:
            baseline_f1 = f1_score(baseline_actuals, baseline_preds)
            baseline_roc_auc = roc_auc_score(baseline_actuals, baseline_preds)

            f1_drift = (baseline_f1 - current_f1) / baseline_f1
            roc_drift = (baseline_roc_auc - current_roc_auc) / baseline_roc_auc

            if abs(f1_drift) > self.alert_threshold or abs(roc_drift) > self.alert_threshold:
                alert_msg = f"ALERT: Performance drift detected! F1 drift: {f1_drift:.3f}, ROC-AUC drift: {roc_drift:.3f}"
                logging.warning(alert_msg)
                return alert_msg

        self.performance_history.append({
            'timestamp': datetime.now(),
            'f1_score': current_f1,
            'roc_auc': current_roc_auc,
            'num_predictions': len(recent_predictions)
        })

        return f"Current F1: {current_f1:.4f}, ROC-AUC: {current_roc_auc:.4f}"

    def check_data_drift(self, reference_data=None):
        """Check for data drift using statistical tests"""
        if len(self.prediction_log) < 100 or reference_data is None:
            return "Insufficient data for data drift detection"

        recent_features = pd.DataFrame([p['features'] for p in self.prediction_log[-500:]])
        reference_df = pd.DataFrame(reference_data)

        drift_detected = False
        drift_features = []

        for col in recent_features.columns:
            if col in reference_df.columns:
                # Simple statistical drift detection using mean difference
                recent_mean = recent_features[col].mean()
                reference_mean = reference_df[col].mean()
                recent_std = recent_features[col].std()
                reference_std = reference_df[col].std()

                # Check if means differ by more than 2 standard deviations
                if abs(recent_mean - reference_mean) > 2 * reference_std:
                    drift_detected = True
                    drift_features.append(col)

        if drift_detected:
            alert_msg = f"ALERT: Data drift detected in features: {drift_features}"
            logging.warning(alert_msg)
            return alert_msg

        return "No significant data drift detected"

    def generate_monitoring_report(self):
        """Generate a comprehensive monitoring report"""
        if not self.performance_history:
            return "No performance history available"

        report = "=== Fraud Detection Monitoring Report ===\n"
        report += f"Total predictions logged: {len(self.prediction_log)}\n"
        report += f"Performance history points: {len(self.performance_history)}\n\n"

        if self.performance_history:
            latest_perf = self.performance_history[-1]
            report += f"Latest Performance Metrics:\n"
            report += f"F1 Score: {latest_perf['f1_score']:.4f}\n"
            report += f"ROC-AUC: {latest_perf['roc_auc']:.4f}\n"
            report += f"Timestamp: {latest_perf['timestamp']}\n\n"

        # Prediction distribution
        if self.prediction_log:
            predictions = [p['prediction'] for p in self.prediction_log]
            fraud_rate = sum(predictions) / len(predictions)
            report += f"Current fraud detection rate: {fraud_rate:.4f}\n\n"

        # Recent alerts
        report += "Recent Alerts:\n"
        with open('fraud_monitoring.log', 'r') as f:
            lines = f.readlines()[-10:]  # Last 10 log entries
            for line in lines:
                if 'ALERT' in line or 'WARNING' in line:
                    report += line

        return report

    def plot_performance_trends(self, save_path='performance_trends.png'):
        """Plot performance trends over time"""
        if not self.performance_history:
            return "No performance data to plot"

        df_perf = pd.DataFrame(self.performance_history)
        df_perf['timestamp'] = pd.to_datetime(df_perf['timestamp'])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # F1 Score trend
        ax1.plot(df_perf['timestamp'], df_perf['f1_score'], marker='o', label='F1 Score')
        ax1.set_title('F1 Score Trend Over Time')
        ax1.set_ylabel('F1 Score')
        ax1.grid(True)
        ax1.legend()

        # ROC-AUC trend
        ax2.plot(df_perf['timestamp'], df_perf['roc_auc'], marker='s', color='orange', label='ROC-AUC')
        ax2.set_title('ROC-AUC Trend Over Time')
        ax2.set_ylabel('ROC-AUC')
        ax2.set_xlabel('Time')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        return f"Performance trends plot saved to {save_path}"

    def plot_prediction_distribution(self, save_path='prediction_distribution.png'):
        """Plot prediction distribution over time"""
        if not self.prediction_log:
            return "No prediction data to plot"

        df_preds = pd.DataFrame(self.prediction_log)
        df_preds['timestamp'] = pd.to_datetime(df_preds['timestamp'])

        # Resample to hourly predictions
        df_preds.set_index('timestamp', inplace=True)
        hourly_fraud_rate = df_preds.resample('H')['prediction'].mean()

        plt.figure(figsize=(12, 6))
        plt.plot(hourly_fraud_rate.index, hourly_fraud_rate.values, marker='o')
        plt.title('Fraud Detection Rate Over Time (Hourly)')
        plt.ylabel('Fraud Rate')
        plt.xlabel('Time')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        return f"Prediction distribution plot saved to {save_path}"

# Global monitor instance
monitor = FraudMonitor()

def get_monitor():
    """Get the global monitor instance"""
    return monitor

if __name__ == "__main__":
    # Example usage
    monitor = FraudMonitor()

    # Simulate some predictions
    for i in range(200):
        features = {f'V{j}': np.random.normal(0, 1) for j in range(1, 29)}
        features['Time'] = np.random.uniform(0, 172800)
        features['Amount'] = np.random.exponential(100)

        prediction = np.random.choice([0, 1], p=[0.995, 0.005])
        probability = np.random.uniform(0, 1)
        actual = np.random.choice([0, 1], p=[0.995, 0.005])

        monitor.log_prediction(features, prediction, probability, actual)

    # Check for drift
    print(monitor.check_performance_drift())
    print(monitor.generate_monitoring_report())
    print(monitor.plot_performance_trends())
    print(monitor.plot_prediction_distribution())
