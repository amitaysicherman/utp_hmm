from Levenshtein import editops
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

EPS = 1e-4


@dataclass
class PhonemesToClustersOps:
    insert: int = 0
    delete: int = 0
    replace: int = 0
    equal: int = 0

    def normalize(self):
        tot = self.insert + self.delete + self.replace + self.equal
        self.insert /= tot
        self.delete /= tot
        self.replace /= tot
        self.equal /= tot

    def from_phonemes_clusters(self, phonemes, clusters):
        for x, _, _ in editops(phonemes, clusters):
            if x == 'insert':
                self.insert += 1
            elif x == 'delete':
                self.delete += 1
            elif x == 'replace':
                self.replace += 1
            else:
                raise Exception(f'Unknown editop {x}')
        self.equal = len(clusters) - self.insert - self.delete - self.replace
        self.normalize()

    def sum(self):
        return self.insert + self.delete + self.replace + self.equal

    def __add__(self, other):

        if self.sum() == 0:
            return other
        if other.sum() == 0:
            return self

        assert abs(self.sum() - 1) < EPS and abs(other.sum() - 1) < EPS
        result = PhonemesToClustersOps(self.insert + other.insert, self.delete + other.delete,
                                       self.replace + other.replace, self.equal + other.equal)
        result.normalize()
        return result

    def __radd__(self, other):
        return self + other

    def __repr__(self):
        return f"insert={self.insert}\ndelete={self.delete}\nreplace={self.replace}\nequal={self.equal})"


with open("models/clusters_phonemes_map_200.txt", "r") as f:
    clusters_to_phonemes = f.read().splitlines()
clusters_to_phonemes = [int(x) for x in clusters_to_phonemes]
clusters_to_phonemes = np.array(clusters_to_phonemes)

with open("data/LIBRISPEECH_TRAIN_clusters_200.txt", "r") as f:
    clusters_ = f.read().splitlines()
clusters = []
for c in tqdm(clusters_):
    c = [int(x) for x in c.split()]
    c = [clusters_to_phonemes[x] for x in c]
    c = [x for x in c if x != 40]
    c = [c[0]] + [c[i] for i in range(1, len(c)) if c[i] != c[i - 1]]
    c = [chr(x + ord("A")) for x in c]
    c = "".join(c)
    clusters.append(c)

with open("data/LIBRISPEECH_TRAIN_idx.txt", "r") as f:
    phonemes = f.read().splitlines()
phonemes = ["".join([chr(int(x) + ord("A")) for x in y.split()]) for y in phonemes]

ops = PhonemesToClustersOps()
for i in range(len(clusters)):
    new_ops = PhonemesToClustersOps()
    new_ops.from_phonemes_clusters(phonemes[i], clusters[i])
    ops = ops + new_ops
print(ops)
