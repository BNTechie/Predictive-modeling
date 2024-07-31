Let's go through a step-by-step example of using an API to serve a machine learning prediction model. We'll use FastAPI to create a RESTful API for a prediction model, and for the example, we'll use a simple Iris flower classification model.

### Step-by-Step Guide

#### Step 1: Train and Save the Model
First, let's train a simple machine learning model using the Iris dataset and save the trained model to a file.

Install Required Libraries

```
pip install pandas scikit-learn joblib

```
Train and Save the Model

Create a script train_model.py:

```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'iris_model.pkl')

print("Model trained and saved successfully!")

```
Run the script to train and save the model:

```
python train_model.py

```

### Step 2: Create a FastAPI Application
Next, we'll create a FastAPI application to serve the trained model.

Install FastAPI and Uvicorn

```
pip install fastapi uvicorn pydantic joblib

```


Create the FastAPI Application

Create a file named app.py:

```
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

```

Run the FastAPI Application
```
uvicorn app:app --reload

```
