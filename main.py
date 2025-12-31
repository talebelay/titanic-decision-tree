from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("decision_tree_model.joblib")

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html") as f:
        return f.read()

@app.post("/predict")
async def predict(pclass: int = Form(...), sex: int = Form(...), age: float = Form(...), 
                  sibsp: int = Form(...), parch: int = Form(...), fare: float = Form(...), 
                  embarked: int = Form(...)):
    
    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    
    # Get the 0 or 1 prediction
    prediction = model.predict(features)[0]
    
    # Get the probability (e.g., [0.2, 0.8])
    probability = model.predict_proba(features)[0]
    confidence = probability[1] if prediction == 1 else probability[0]
    
    result = "Survived" if prediction == 1 else "Did Not Survive"
    
    return {
        "prediction": result,
        "confidence": f"{round(confidence * 100, 2)}%"
    }
