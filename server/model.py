from imports import *

"""
This class represents a Model.
To run make a new model class then call construct_model to make a new model.

"""

class Model:
    model = None

    # Constructor, creates the model
    def __init__(self):
        self.model = None

    # Returns a new BERT model using the emotion_set
    def construct_model(self):
        # Import the training & validation csv
        df = pd.read_csv('data/goemotions_1.csv')

        

        # Return the model
        return None
    
model = Model()
model.construct_model()