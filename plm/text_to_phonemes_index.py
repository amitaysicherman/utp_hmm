import random

import numpy as np
from g2p_en import G2p
from tqdm import tqdm
import argparse
from mapping import phonemes_to_index

MAX_P = max(phonemes_to_index.values())


def raplace_random(x, p):
    x = x.copy()
    for i in range(len(x)):
        if np.random.rand() < p:
            x[i] = np.random.randint(0, MAX_P)
    return x


def add_noise_to_file(file_name, output_base, g2p=None, repeats=50):
    print("start reading", file_name)
    with open(file_name, 'r') as file:
        lines = file.read().splitlines()
    all_phonemes = []
    for line in tqdm(lines):
        if g2p:
            sentence = " ".join(line.split(' ')[1:])
            phonemes = g2p(sentence)
            phonemes = [p[:-1] if p[-1].isnumeric() else p for p in phonemes]
            phonemes = [p for p in phonemes if p != "'"]
            phonemes = [p for p in phonemes if p != " "]
        else:
            phonemes = line.split(' ')
        phonemes = [phonemes_to_index[p] for p in phonemes if p in phonemes_to_index]
        all_phonemes.append(np.array(phonemes))
    final_clean = []
    final_noise = []
    print("start adding noise")
    for i in tqdm(range(repeats)):
        for phonemes in all_phonemes:
            final_clean.append(" ".join([str(p) for p in phonemes]))
            noise_phones = raplace_random(phonemes, random.random())
            final_noise.append(" ".join([str(p) for p in noise_phones]))
    with open(output_base + "_clean.txt", 'w') as file:
        file.write("\n".join(final_clean))
    with open(output_base + "_noise.txt", 'w') as file:
        file.write("\n".join(final_noise))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['tm', 'lr'], default="tm")

    vars = parser.parse_args()
    if vars.dataset == "tm":
        train_file = "TIMIT_TRAIN_PH.txt"
        valid_output_file = "TIMIT_TRAIN_VAL_PH"
        test_file = "TIMIT_TEST_PH.txt"
        g2p = None
        repeats = 500
    else:  # 'lr'
        train_file = "libri_train.txt"
        valid_output_file = "libri_train_val"
        test_file = "libri_test.txt"
        g2p = G2p()
        repeats = 10

    add_noise_to_file(train_file, output_base=train_file.replace(".txt", ""), g2p=g2p, repeats=repeats)
    add_noise_to_file(train_file, output_base=valid_output_file, g2p=g2p, repeats=repeats)
    add_noise_to_file(test_file, output_base=test_file.replace(".txt", ""), g2p=g2p, repeats=repeats)
