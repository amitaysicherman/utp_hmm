import joblib
from sklearn.cluster import KMeans
# import PCA
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt

with open("models/clusters_phonemes_map_100.txt", "r") as f:
    clusters_to_phonemes = f.read().splitlines()
clusters_to_phonemes = [int(x) for x in clusters_to_phonemes]
real_vad = np.array(clusters_to_phonemes) == 40
clusters_to_phonemes = np.array(['r' if c == 40 else 'g' for c in clusters_to_phonemes])

km = joblib.load("models/km100.bin")
km_2d = PCA(n_components=2).fit_transform(km.cluster_centers_)
vad_clusters = KMeans(n_clusters=2).fit_predict(km_2d)

mask = (vad_clusters == 1) & (real_vad == 1)
v = mask.sum()
plt.scatter(km_2d[mask, 0], km_2d[mask, 1], marker='o', c='r', label=f"True Positive ({v})")

mask = (vad_clusters == 1) & (real_vad == 0)
v = mask.sum()
plt.scatter(km_2d[mask, 0], km_2d[mask, 1], marker='o', c='b', label=f"False Positive ({v})")

mask = (vad_clusters == 0) & (real_vad == 1)
v = mask.sum()
plt.scatter(km_2d[mask, 0], km_2d[mask, 1], marker='o', c='y', label=f"False Negative ({v})")

mask = (vad_clusters == 0) & (real_vad == 0)
v = mask.sum()
plt.scatter(km_2d[mask, 0], km_2d[mask, 1], marker='o', c='g', label=f"True Negative ({v})")
plt.legend()

# plt.scatter(km_2d[:, 0], km_2d[:, 1], c=vad_clusters, marker='o', s=50, edgecolors=clusters_to_phonemes)
# plt.title(f'{(vad_clusters[vad_clusters == 1] == real_vad[vad_clusters == 1]).mean():.0%}({np.sum(vad_clusters)})')
plt.savefig("vad_clusters.png", dpi=500)
plt.show()
