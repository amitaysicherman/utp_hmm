import random

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from train import max_len, input_size, d_model, num_layers, nhead, PhonemesDataset, mask_value, padding_value
from x_transformers import TransformerWrapper, Encoder
import torch
from train import get_model

cp_file = "./models/best.cp"
data_path = 'LR960_PH.npz'
data_len_path = 'LR960_PH_LEN.txt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model()
model.load_state_dict(torch.load(cp_file, map_location=torch.device('cpu')))

model = model.to(device)

dataset = PhonemesDataset(data_path=data_path, data_len_path=data_len_path)

# x_to_layer = {}
tot_max = 10_000_000

features = []
labels = []

model.eval()
with torch.no_grad():
    while True:
        print(f'{len(labels):,}', flush=True)
        i = random.randint(0, len(dataset) - 1)
        x = dataset[i]
        x = x.unsqueeze(0)
        x = x.to(device)
        output = model(x, return_intermediates=True, return_attn=True)
        layers = output[1].hiddens
        labels_ = x[0].detach().cpu().numpy()
        features_ = layers[5][0].detach().cpu().numpy()
        features_ = features_[labels_ != padding_value]
        labels_ = labels_[labels_ != padding_value]
        features.append(features_)
        labels.extend(list(labels_))

        if len(labels) > tot_max:
            break
features = np.vstack(features)
labels = np.array(labels)
for k in [50, 100, 200, 500, 1000, 2000]:
    # kmeans = KMeans(n_clusters=k, random_state=0).fit(features)

    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, verbose=10, batch_size=10_000, init_size=10_000).fit(
        features)
    k_means_labels = kmeans.labels_
    err = 0
    for v in range(k):
        counts = np.unique(labels[k_means_labels == v], return_counts=True)[1]
        err += counts.sum() - counts.max()
    err = err / len(labels)
    print(f"K={k}, err={err}", flush=True)
    np.save(f"kmeans_10M_{k}.npy", kmeans.cluster_centers_)
