import joblib
import pandas as pd

model = joblib.load('models/trained_model.pkl')

def detect_anomalies(log_data):
    df = pd.DataFrame(log_data)
    predictions = model.predict(df)
    return predictions
