from imports import *
import pickle
from model import Model
from model import BertClassifier
from model import loadModel

# Load in the config file.
config = toml.load('config.toml')

# Load in the model
model_path = config['server']['model_path']

# Step 1: Load the PyTorch model from the .pth file
model = loadModel(config['server']['model_path'])

# Step 2: Serialize the model using pickle
serialized_model = pickle.dumps(model)

# Step 3: Save the serialized model to a .pkl file
with open('spotify-emotions-model.pkl', 'wb') as f:
    f.write(serialized_model)
