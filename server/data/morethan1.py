import pandas as pd
import numpy as np

path = "testemotions_1.csv"
out_path = "testemotions_1_fixed.csv"
df = pd.read_csv(path)

to_drop = []
for index, row in df.iterrows():
    arr = row[-28:]
    numpy_array = arr.values

    if np.sum(numpy_array == 1) > 1:
        to_drop.append(index)

df = df.drop(to_drop)

for index, row in df.iterrows():
    arr = row[-28:]
    numpy_array = arr.values

    if np.sum(numpy_array == 1) > 1:
        print(f"issue {index}")

df.to_csv(out_path, index=False)