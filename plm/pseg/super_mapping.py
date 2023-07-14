import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

phonemes_to_index = {'AA': 0, 'AE': 1, 'AH': 2, 'AO': 3, 'AW': 4, 'AY': 5, 'B': 6, 'CH': 7, 'D': 8, 'DH': 9, 'EH': 10,
                     'ER': 11, 'EY': 12, 'F': 13, 'G': 14, 'HH': 15, 'IH': 16, 'IY': 17, 'JH': 18, 'K': 19, 'L': 20,
                     'M': 21, 'N': 22, 'NG': 23, 'OW': 24, 'OY': 25, 'P': 26, 'R': 27, 'S': 28, 'SH': 29, 'T': 30,
                     'TH': 31, 'UH': 32, 'UW': 33, 'V': 34, 'W': 35, 'Y': 36, 'Z': 37, 'ZH': 38}
index_to_phonemes = {v: k for k, v in phonemes_to_index.items()}

with open("./data/p_superv/features.phonemes", 'r') as f:
    phonemes_data = f.readlines()
phonemes_data = [[phonemes_to_index[x.upper()] if x.upper() != "DX" else phonemes_to_index["T"] for x in
                  line.strip().split()] for line in phonemes_data]
mapping = np.zeros((39, 100))
with open("./data/p_superv/features.clusters", 'r') as f:
    clusters_data = f.readlines()
clusters_data = [[int(x) for x in line.strip().split()] for line in clusters_data]

for p_line, c_line in zip(phonemes_data, clusters_data):
    for p, c in zip(p_line, c_line):
        mapping[p, c] += 1
mapping = pd.DataFrame(mapping, index=[index_to_phonemes[i] for i in range(39)])

mapping_norm = mapping.div(mapping.sum(axis=1), axis=0)

for i in range(100):
    print(i)
    s_values = sorted(mapping[i].values)[::-1]
    fig = plt.figure(figsize=(7, 7))
    plt.bar(range(39), s_values)
    plt.title(f'{i} {mapping[i].sum()}')
    plt.show()

for i in range(39):
    print(i)
    s_values = sorted(mapping_norm.iloc[i].values)[::-1]
    fig = plt.figure(figsize=(7, 7))
    plt.bar(range(100), s_values)
    plt.title(f'{index_to_phonemes[i]} {mapping.iloc[i].sum()}')
    plt.show()

    # plt.close(fig)
