import random

import numpy as np
from g2p_en import G2p
from tqdm import tqdm
import argparse
from mapping import phonemes_to_index, mis_index

MAX_P = max(phonemes_to_index.values())


def raplace_random(x, p):
    x = x.copy()
    for i in range(len(x)):
        if np.random.rand() < p:
            x[i] = np.random.randint(0, MAX_P)
    return x


def replace_random_with_mis(x, p):
    x = x.copy()
    y = x.copy()
    add_count = int(len(x) * p)
    for _ in range(add_count):
        i = np.random.randint(0, len(x))
        v = np.random.randint(0, MAX_P)
        x = np.insert(x, i, mis_index)
        y = np.insert(y, i, v)
    return x, y


def add_noise_to_file(file_name, output_base, g2p=None, repeats=50, mis_prob=0.2):
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

        # phonemes = [phonemes_to_index[p] for p in phonemes if p in phonemes_to_index]

        all_phonemes.append(np.array(phonemes))
    if "val" in output_base:
        all_phonemes=random.choices(all_phonemes,k=10000)
    final_clean = []
    final_noise = []
    print("start adding noise")
    for i in tqdm(range(repeats)):
        for phonemes in all_phonemes:
            if mis_prob > 0:
                clean_with_mis, noise_with_mis = replace_random_with_mis(phonemes, mis_prob)
            else:
                clean_with_mis = phonemes
                noise_with_mis = phonemes
            final_clean.append(" ".join([str(p) for p in clean_with_mis]))
            noise_phones = raplace_random(noise_with_mis, random.random())

            final_noise.append(" ".join([str(p) for p in noise_phones]))
    mis_name = "_mis" if mis_prob > 0 else ""
    with open(output_base + f"{mis_name}_clean.txt", 'w') as file:
        file.write("\n".join(final_clean))
    with open(output_base + f"{mis_name}_noise.txt", 'w') as file:
        file.write("\n".join(final_noise))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['tm', 'lr'], default="lr")
    parser.add_argument('--add_mis', type=float, default=0.0)

    vars = parser.parse_args()
    if vars.dataset == "tm":
        train_file = "TIMIT_TRAIN_PH.txt"
        valid_output_file = "TIMIT_TRAIN_VAL_PH"
        test_file = "TIMIT_TEST_PH.txt"
        g2p = None
        repeats = 500
    else:  # 'lr'
        train_file = "lr_train.txt"
        valid_output_file = "lr_train_val.txt"
        test_file = "lr_test.txt"
        g2p = None# G2p()
        repeats = 10

    add_noise_to_file(train_file, output_base=train_file.replace(".txt", ""), g2p=g2p, repeats=repeats,
                      mis_prob=vars.add_mis)
    add_noise_to_file(train_file, output_base=valid_output_file, g2p=g2p, repeats=repeats, mis_prob=vars.add_mis)
    add_noise_to_file(test_file, output_base=test_file.replace(".txt", ""), g2p=g2p, repeats=repeats,
                      mis_prob=vars.add_mis)
