from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from monitoring import get_monitor

app = FastAPI(title="Fraud Detection API", description="API for detecting credit card fraud", version="1.0.0")

# Load the trained model (assuming we save it later)
# model = pickle.load(open('model.pkl', 'rb'))
# scaler = pickle.load(open('scaler.pkl', 'rb'))

# Initialize monitor
monitor = get_monitor()

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        # Convert to array
        features = np.array([[
            transaction.Time, transaction.V1, transaction.V2, transaction.V3, transaction.V4,
            transaction.V5, transaction.V6, transaction.V7, transaction.V8, transaction.V9,
            transaction.V10, transaction.V11, transaction.V12, transaction.V13, transaction.V14,
            transaction.V15, transaction.V16, transaction.V17, transaction.V18, transaction.V19,
            transaction.V20, transaction.V21, transaction.V22, transaction.V23, transaction.V24,
            transaction.V25, transaction.V26, transaction.V27, transaction.V28, transaction.Amount
        ]])

        # Scale features
        # features_scaled = scaler.transform(features)

        # Make prediction
        # prediction = model.predict(features_scaled)[0]
        # probability = model.predict_proba(features_scaled)[0][1]

        # For now, return dummy response
        prediction = 0  # 0: normal, 1: fraud
        probability = 0.05

        # Log prediction for monitoring
        feature_dict = {
            'Time': transaction.Time,
            'V1': transaction.V1, 'V2': transaction.V2, 'V3': transaction.V3, 'V4': transaction.V4,
            'V5': transaction.V5, 'V6': transaction.V6, 'V7': transaction.V7, 'V8': transaction.V8,
            'V9': transaction.V9, 'V10': transaction.V10, 'V11': transaction.V11, 'V12': transaction.V12,
            'V13': transaction.V13, 'V14': transaction.V14, 'V15': transaction.V15, 'V16': transaction.V16,
            'V17': transaction.V17, 'V18': transaction.V18, 'V19': transaction.V19, 'V20': transaction.V20,
            'V21': transaction.V21, 'V22': transaction.V22, 'V23': transaction.V23, 'V24': transaction.V24,
            'V25': transaction.V25, 'V26': transaction.V26, 'V27': transaction.V27, 'V28': transaction.V28,
            'Amount': transaction.Amount
        }
        monitor.log_prediction(feature_dict, prediction, probability)

        return {
            "prediction": int(prediction),
            "fraud_probability": float(probability),
            "message": "Fraud detected" if prediction == 1 else "Transaction appears normal"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is running"}

@app.get("/monitoring/report")
def get_monitoring_report():
    """Get monitoring report"""
    return {"report": monitor.generate_monitoring_report()}

@app.get("/monitoring/performance")
def get_performance_drift():
    """Check for performance drift"""
    return {"drift_check": monitor.check_performance_drift()}

@app.post("/monitoring/feedback")
def add_feedback(prediction_id: int, actual: int):
    """Add feedback for a prediction (actual label)"""
    # In a real implementation, you'd need to match prediction_id to logged predictions
    # For now, just log the feedback
    monitor.prediction_log[-1]['actual'] = actual  # Simple implementation
    return {"message": "Feedback recorded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
