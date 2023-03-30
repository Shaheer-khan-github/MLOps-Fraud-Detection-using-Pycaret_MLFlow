# **MLOps-Fraud-Detection-using-Pycaret_MLFlow**

 PyCaret is a low-code machine learning library that allows you to quickly build and train models, while MLflow provides a robust platform for managing the entire machine learning lifecycle.
 Together, these tools make it easy to develop and deploy fraud detection models with confidence. By using PyCaret's built-in functionalities for data preprocessing, feature engineering, model training, and hyperparameter tuning, you can quickly build high-performing models that are optimized for fraud detection. And with MLflow's tracking and versioning capabilities, you can easily manage your experiments, reproduce your results, and collaborate with your team.


#### [fraud_voting_model_api.py](fraud_voting_model_api.py)
This is a Python script for deploying a fraud detection model using FastAPI and Pycaret. The script loads a pre-trained model, defines a predict function, and runs a FastAPI app to make predictions on incoming data.


#### [model_utils.py](model_utils.py)
The code defines a function called get_raw_data() that reads a CSV file containing credit card data, performs undersampling, and splits the data into training and testing sets. It then returns the resulting training and testing dataframes.