import os
import sys
import pandas as pd
import numpy as np
import dill 
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(obj, file_path):
    '''
    This function is used to save the object as a pickle file.
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except CustomException as e:
        raise CustomException(e,sys)
    
def evaluate_models(models,X_train,y_train,X_test,y_test,params):
    '''
    This function is used to evaluate the models based on the training and testing data.
    '''
    try:
        model_report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            # model.fit(X_train,y_train)      # Training model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            model_report[list(models)[i]] = test_model_score
        
        return model_report
    
    except CustomException as e:
        raise CustomException(e,sys)

def load_object(file_path):
    '''
    This function is used to load the object from the pickle file.
    '''
    try:
        with open(file_path, 'rb') as file:
            obj = dill.load(file)
            return obj
    except CustomException as e:
        raise CustomException(e,sys)