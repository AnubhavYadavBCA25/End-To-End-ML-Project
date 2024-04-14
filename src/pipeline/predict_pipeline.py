import pandas as pd
import sys
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path = "artifact\model.pkl"
            preprocessor_pth = "artifact\preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_pth)
            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            return prediction
        except CustomException as e:
            raise CustomException(e,sys)

class CustomeData:
    def __init__(self,
        gender: str,
        race_ethnicity:str,
        parental_level_of_education:str,
        lunch:str,
        test_preparation_course:str,
        reading_score:int,
        writing_score:int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_df(self):
        try:
            custome_input_data = {
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
            }
            return pd.DataFrame(custome_input_data)
        
        except CustomException as e:
            raise CustomException(e,sys)
