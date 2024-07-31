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
### Step 3: Test the API

You can test the API using curl, Postman, or any other HTTP client.

Using curl:

```
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

```

Using Postman:

Open Postman and create a new POST request.

Set the URL to http://localhost:8000/predict.

Set the Content-Type to application/json.

In the body, select raw and enter the JSON payload:

json

```
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

```
### Step 4: Create a Docker Container
To containerize the FastAPI application, create a Dockerfile.

Create requirements.txt

Create a requirements.txt file with your dependencies:

```
fastapi
uvicorn
pydantic
joblib
scikit-learn
numpy
```

Create Dockerfile

Create a Dockerfile in the same directory as your app.py and requirements.txt:

```
# Use the official Python image from the Docker Hub
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]

```
Build the Docker Image

Open a terminal, navigate to the directory containing your Dockerfile, and run:
