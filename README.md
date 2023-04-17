# **MLOps-Fraud-Detection-using-Pycaret_MLFlow**

![Credit Card Fraud](Credit%20card_fraud.jpg)

 PyCaret is a low-code machine learning library that allows you to quickly build and train models, while MLflow provides a robust platform for managing the entire machine learning lifecycle.
 Together, these tools make it easy to develop and deploy fraud detection mPodels with confidence. By using PyCaret's built-in functionalities for data preprocessing, feature engineering, model training, and hyperparameter tuning, you can quickly build high-performing models that are optimized for fraud detection. And with MLflow's tracking and versioning capabilities, you can easily manage your experiments, reproduce your results, and collaborate with your team.

[1. Model_Training.ipynb](1.%20Model_Training.ipynb)
contains trainings with different processing steps like fixing imbalance, removing outliers, and feature engineering. It also contains the Experiment Tracking with MLflow.

[2. Train_and_Evaluate_the_Model.ipynb](2.%20Train_and_Evaluate_the_Model.ipynb)
contains the training of the selected model and evaluation of the best classifier.

[3. Optimize_the_Model](3.%20Optimize_the_Model.ipynb)
contains the code for hyper parameter tuning and voting classifier then created REST API for the model using FastAPI and create a Docker [image](Dockerfile) for the API.

[fraud_voting_model_api.py](fraud_voting_model_api.py)
This script defines a FastAPI app that loads a pre-trained fraud detection model, and provides two endpoints (Post and get request) to make predictions on new data. The endpoints accept 30 input features related to a financial transaction and return the model's binary classification prediction of fraud or not fraud. The app is run locally on port 8001 using uvicorn. The weight file of `fraud_voting_model_api.py` is [Weight](fraud_voting_model_api.pkl)

[model_utils.py](model_utils.py) 
The code defines a function called get_raw_data() that reads a CSV file containing credit card data, performs undersampling, and splits the data into training and testing sets. It then returns the resulting training and testing dataframes.

