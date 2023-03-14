import pandas as pd
from pycaret.classification import *
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import mlflow
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import average_precision_score
from sklearn.utils import shuffle

def get_raw_data():
    df = pd.read_csv('dataset' + os.sep + 'creditcard.csv', encoding_errors='ignore', on_bad_lines='skip')
    df = shuffle(df, random_state=5)
    df_0 = df.loc[df.Class == 0].iloc[0:int((len(df)/100))] # increasing the calculation speed, some undersampling - drop random records with 0 targets in order to optimize the calculation time
    df_1 = df.loc[df.Class == 1]
    df = pd.concat([df_0, df_1])    
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Class']), df['Class'], test_size=0.33, random_state=42)
    X_train['target'] = y_train
    X_test['target'] = y_test
    df_train = X_train.copy()
    df_test = X_test.copy()
    return df_train, df_test
"""Get raw data
            Parameters
            ----------
            nothing
            
            Returns
            train dataset and test dataset
            -------
            """
        