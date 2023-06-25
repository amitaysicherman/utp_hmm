# sbatch --gres=gpu:1,vmem:24g --mem=75G --time=7-0 --wrap "python timit_kmeans.py"
import argparse
import os.path as osp

import numpy as np
import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def learn_kmeans(feats, nclusters, niter=500, nredo=10, output_file=None, return_centers=True):
    kmeans = KMeans(n_clusters=nclusters, n_init=nredo, max_iter=niter, verbose=10, random_state=42)
    kmeans.fit(feats)
    centroids = kmeans.cluster_centers_
    if output_file is not None:
        np.save(output_file, centroids)
    if return_centers:
        return centroids


def apply_kmeans(centroids, lens, features, output_file=None, return_clusters=False):
    pred_cluster = []

    curr = 0
    for l in tqdm.tqdm(lens):
        f = features[curr:curr + l]
        curr += l
        clusters = pairwise_distances(f, centroids).argmin(axis=-1)
        pred_cluster.append(" ".join(str(x) for x in clusters))

    if output_file:
        with open(output_file, 'w') as f:
            f.write("\n".join(pred_cluster))
    if return_clusters:
        return pred_cluster


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nclusters",
        help="number of clusters",
        default=100,
        type=int
    )
    parser.add_argument(
        "--force_learning",
        default=1,
        type=int
    )
    parser.add_argument(
        "--save_dir",
        help="save_dir",
        default="data",
    )
    args = parser.parse_args()
    base_dir = args.save_dir
    with open(osp.join(base_dir, "features.length")) as f:
        lens = [int(x) for x in f.read().splitlines()]
    features = np.load(osp.join(base_dir, "features.npy"))

    c_path = osp.join(base_dir, f"centroids_{args.nclusters}.npy")
    if not osp.exists(c_path) or bool(args.force_learning):
        centroids = learn_kmeans(features, args.nclusters, output_file=c_path)
    else:
        centroids = np.load(c_path)

    label_path = osp.join(base_dir, f"clusters_{args.nclusters}.txt")
    apply_kmeans(centroids, lens, features, output_file=label_path)


if __name__ == "__main__":
    main()
