from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load('iris_model.pkl')

# Create a FastAPI instance
app = FastAPI()

# Define the input data model
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define the prediction endpoint
@app.post("/predict")
def predict(iris: IrisInput):
    data = np.array([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}

