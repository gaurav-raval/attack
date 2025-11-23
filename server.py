import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# ================================
# Load Your Models
# ================================
sql_vectorizer = joblib.load("SQLInjection_vectorizer.pkl")
sql_model = joblib.load("SQLInjection_Model.pkl")

xss_vectorizer = joblib.load("xss_vectorizer.pkl")
xss_model = joblib.load("xss_sqli_model.pkl")

# ================================
# Request Schema
# ================================
class Payload(BaseModel):
    payload: str

# ================================
# Safe Conversion Helper
# ================================
def to_python(val):
    """Convert numpy types to native Python types."""
    if isinstance(val, (np.int64, np.int32, np.int16, np.int8)):
        return int(val)
    if isinstance(val, (np.float64, np.float32, np.float16)):
        return float(val)
    if isinstance(val, (np.bool_)):
        return bool(val)
    return val


# ================================
# Prediction Function
# ================================
def run_model(model, vectorizer, text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    pred = to_python(pred)  # convert numpy to Python

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec)[0].max()
        prob = to_python(prob)  # convert numpy to Python
    else:
        prob = None

    return pred, prob


# ================================
# FastAPI App
# ================================
app = FastAPI()

@app.post("/analyze")
def analyze(data: Payload):
    user_input = data.payload

    sql_pred, sql_conf = run_model(sql_model, sql_vectorizer, user_input)
    xss_pred, xss_conf = run_model(xss_model, xss_vectorizer, user_input)

    return {
        "input": user_input,
        "sql_model": {
            "prediction": sql_pred,
            "confidence": sql_conf
        },
        "xss_model": {
            "prediction": xss_pred,
            "confidence": xss_conf
        }
    }
