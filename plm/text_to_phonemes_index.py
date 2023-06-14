import random

import numpy as np
from g2p_en import G2p
from tqdm import tqdm
import argparse

phonemes_to_index = {'AA': 0, 'AE': 1, 'AH': 2, 'AO': 3, 'AW': 4, 'AY': 5, 'B': 6, 'CH': 7, 'D': 8, 'DH': 9, 'EH': 10,
                     'ER': 11, 'EY': 12, 'F': 13, 'G': 14, 'HH': 15, 'IH': 16, 'IY': 17, 'JH': 18, 'K': 19, 'L': 20,
                     'M': 21, 'N': 22, 'NG': 23, 'OW': 24, 'OY': 25, 'P': 26, 'R': 27, 'S': 28, 'SH': 29, 'T': 30,
                     'TH': 31, 'UH': 32, 'UW': 33, 'V': 34, 'W': 35, 'Y': 36, 'Z': 37, 'ZH': 38}
MAX_P = max(phonemes_to_index.values())


def raplace_random(x, p):
    x = x.copy()
    for i in range(len(x)):
        if np.random.rand() < p:
            x[i] = np.random.randint(0, 39)
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="libri_test.txt")
    parser.add_argument('--output_prefix', type=str, default="LRTEST")
    parser.add_argument('--do_phonemes', type=bool, default=True)
    vars = parser.parse_args()
    g2p = G2p()
    with open(vars.input_file, 'r') as file:
        lines = file.read().splitlines()

    all_phonemes = []
    all_length = []
    for line in tqdm(lines):
        line = line.strip().upper()
        sentence = " ".join(line.split(' ')[1:])  # Extract the sentence after the tab character
        phonemes = g2p(sentence)
        phonemes = [p[:-1] if p[-1].isnumeric() else p for p in phonemes]
        phonemes = [p for p in phonemes if p != "'"]
        phonemes = [p for p in phonemes if p != " "]

        all_length.append(len(phonemes))

        for p in phonemes:
            if p not in phonemes_to_index:
                print(p)
        phonemes = [phonemes_to_index[p] for p in phonemes if p in phonemes_to_index]
        all_phonemes.append(np.array(phonemes))

    final_clean = []
    final_noise = []
    for i in range(10):
        for phonemes in all_phonemes:
            final_clean.append(" ".join([str(p) for p in phonemes]))
            noise_phones = raplace_random(phonemes, random.random())
            final_noise.append(" ".join([str(p) for p in noise_phones]))
    with open(f'{vars.output_prefix}_CLEAN.txt', 'w') as file:
        file.write("\n".join(final_clean))
    with open(f'{vars.output_prefix}_NOISE.txt', 'w') as file:
        file.write("\n".join(final_noise))
