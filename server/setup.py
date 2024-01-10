from imports import *
from tensorflow.python.client import device_lib

print(tf.__version__)
print(device_lib.list_local_devices())
print("\n\n\n")
print(tf.config.list_physical_devices('GPU'))