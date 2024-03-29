import joblib
import numpy as np
import os
data_name="p_superv"
input_file = f"data/{data_name}/features.npy"
len_file = f"data/{data_name}/features.length"
output_file = f"data/{data_name}/features.clusters"
km_file = "models/km100.bin"

km = joblib.load(km_file)

features = np.load(input_file)
with open(len_file, 'r') as f:
    lengths = f.read().split("\n")
lengths = [int(l) for l in lengths]
assert sum(lengths) == len(features)
features = np.split(features, np.cumsum(lengths)[:-1])
if os.path.exists(output_file):
    os.remove(output_file)
for f in features:
    clusters = km.predict(f)
    with open(output_file, 'a') as f:
        f.write(" ".join([str(c) for c in clusters]) + "\n")
