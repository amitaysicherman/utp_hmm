import numpy as np
from jiwer import wer
import matplotlib.pyplot as plt
from mapping import index_to_letter

with open("models/clusters_phonemes_map_letters_200.txt") as f:
    clusters_to_phonemes = f.read().splitlines()
clusters_to_phonemes = [int(x) for x in clusters_to_phonemes]
clusters_to_phonemes = np.array(clusters_to_phonemes)
with open("data/LIBRISPEECH_TEST_clusters_200.txt") as f:
    clusters = f.read().splitlines()
clusters = [[int(y) for y in x.split()] for x in clusters]
clusters = np.array(clusters)

with open("data/LIBRISPEECH_TEST_letters.txt") as f:
    letters = f.read().splitlines()
letters = [[int(y) for y in x.split()] for x in letters]
letters = np.array(letters)
assert len(clusters) == len(letters)
scores = []
for c, l in zip(clusters, letters):
    c = [clusters_to_phonemes[x] for x in c if clusters_to_phonemes[x] != 40]
    c = [c[0]] + [c[i] for i in range(1, len(c)) if c[i] != c[i - 1]]
    c = " ".join([str(x) for x in c])
    l = " ".join([str(x) for x in l])
    scores.append(wer(c, l))
    c= "".join([index_to_letter[int(x)] for x in c.split()])
    l= "".join([index_to_letter[int(x)] for x in l.split()])
    print(c)
    print(l)
    # break
print(np.mean(scores))
plt.hist(scores)
plt.show()
