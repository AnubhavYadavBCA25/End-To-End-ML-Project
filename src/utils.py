import os
import sys
import pandas as pd
import numpy as np
import dill 

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