from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
app = FastAPI()

# Load model
model = joblib.load("model/house_price_model.pkl")

@app.get("/")
def home():
    return {"message": "House Price Prediction API Running"}

"""@app.post("/predict")
def predict(features: list):
    data = np.array(features).reshape(1, -1)
    prediction = model.predict(data)
    return {"predicted_price": float(prediction[0])}"""

class HouseInput(BaseModel):
    OverallQual: int
    GrLivArea: float
    GarageCars: int
    TotalBdmtSF: float
    YearBuilt: int
@app.post("/predict")
def predict(input_data: HouseInput):
    features=[
        input_data.OverallQual,
        input_data.GrLivArea,
        input_data.GarageCars,
        input_data.TotalBdmtSF,
        input_data.YearBuilt
        
    ]
    full_features=features+[0]*(247-len(features))

    data = np.array(full_features).reshape(1, -1)
    prediction = model.predict(data)
    return {"predicted_price": float(prediction[0])}