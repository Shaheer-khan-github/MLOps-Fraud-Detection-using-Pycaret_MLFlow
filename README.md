# **MLOps-Fraud-Detection-using-Pycaret_MLFlow**

![Credit Card Fraud](Credit%20card_fraud.jpg)

 PyCaret is a low-code machine learning library that allows you to quickly build and train models, while MLflow provides a robust platform for managing the entire machine learning lifecycle.
 Together, these tools make it easy to develop and deploy fraud detection models with confidence. By using PyCaret's built-in functionalities for data preprocessing, feature engineering, model training, and hyperparameter tuning, you can quickly build high-performing models that are optimized for fraud detection. And with MLflow's tracking and versioning capabilities, you can easily manage your experiments, reproduce your results, and collaborate with your team.

## **Key Tools or Libraries**
1. PyCaret: a low-code machine learning library for building and training models.
2. MLflow: a platform for managing the machine learning lifecycle, including experiment tracking and versioning.
3. FastAPI: a modern, fast (high-performance) web framework for building APIs with Python.
4. Docker: a containerization platform for packaging software into portable, self-contained environments.
5. Uvicorn: a lightning-fast ASGI server for Python, used to run the FastAPI app.

[1. Model_Training.ipynb](1.%20Model_Training.ipynb)
contains trainings with different processing steps like fixing imbalance, removing outliers, and feature engineering. It also contains the Experiment Tracking with MLflow.

[2. Train_and_Evaluate_the_Model.ipynb](2.%20Train_and_Evaluate_the_Model.ipynb)
contains the training of the selected model and evaluation of the best classifier.

[3. Optimize_the_Model](3.%20Optimize_the_Model.ipynb)
contains the code for hyper parameter tuning and voting classifier then created REST API for the model using FastAPI and create a Docker [image](Dockerfile) for the API.

[4. Testing_of_API](4.%20Testing_of_API.ipynb)
This code file uses a trained model for fraud voting detection and deploys it using a Docker container. It loads data from a file, checks the Docker images and starts the API container. It then sends a request to the API endpoint for the prediction of the target variable for a particular record and prints the response. The response contains the predicted target value for the specified record.

[model_utils.py](model_utils.py) 
The code defines a function called get_raw_data() that reads a CSV file containing credit card data, performs undersampling, and splits the data into training and testing sets. It then returns the resulting training and testing dataframes.

[fraud_voting_model_api.py](fraud_voting_model_api.py)
This script defines a FastAPI app that loads a pre-trained fraud detection model, and provides an endpoint (POST request) to make predictions on new data. The endpoints accept 30 input features related to a financial transaction and return the model's binary classification prediction of fraud or not fraud. The app is run locally on port 8001 using uvicorn. The weight file of `fraud_voting_model_api.py` is [Weight](fraud_voting_model_api.pkl)

[fraud_voting_model_api_with_docker](fraud_voting_model_api_with_docker.py)
This Python script defines a FastAPI for deploying a fraud voting model using the PyCaret classification library. It loads a pre-trained pipeline model and defines two endpoints for making predictions: POST /predict and GET /get_predict. The app runs on a Uvicorn server listening on port 8007.

[Dockerfile_8007](Dockerfile_8007)
This is a Dockerfile that defines a Python environment with specific dependencies, installs pycaret and additional packages specified in `requirements2.txt`, and exposes port 8007. The final command runs a Python script for deploying a fraud voting model API. 

`Note:` 
1. The port should be different from the other already existing ports. As you can see the port in 
[Docker file](Dockerfile) is different from [Dockerfile_8007](Dockerfile_8007) and both have different `requirements.txt` files.

2. The [`fraud_voting_model_api_with_docker.py`](fraud_voting_model_api_with_docker.py) file is different to [`fraud_voting_model_api.py`](fraud_voting_model_api.py) file regarding the API endpoints, ports and host IP. 


