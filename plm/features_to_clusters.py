from utils import args_parser
import joblib
from sklearn.cluster import MiniBatchKMeans
import numpy as np

args = args_parser()

features_path = f"{args.match_data}/features.npy"

len_path = f"{args.match_data}/features.length"

with open(len_path, 'r') as f:
    length = f.read().splitlines()
length = [int(x) for x in length]

km: MiniBatchKMeans = joblib.load(args.km_model)

features = np.load(features_path)
clusters = []
cur = 0
for l in length:
    x = features[cur:cur + l]
    c = km.predict(x)
    clusters.append(" ".join([str(xx) for xx in c]))
    cur += l

clusters_file = f"{args.match_data}/features.clusters"
with open(clusters_file, 'w') as f:
    f.write("\n".join(clusters))
