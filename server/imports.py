# Flask
from flask import Flask, jsonify, request
from flask_cors import CORS
from waitress import serve

# Tensor Flow
import tensorflow as tf
import tensorflow_hub as hub

# Numpy and Pandas and torch
import numpy as np
import pandas as pd

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# JSON
import json as json

# Beautiful Soup
import requests
from bs4 import BeautifulSoup

# TOML 
import toml
