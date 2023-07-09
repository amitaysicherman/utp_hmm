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


def replace_random_with_dup(x, y, p):
    x = x.copy()
    y = y.copy()
    add_count = int(len(x) * p)
    for _ in range(add_count):
        i = np.random.randint(1, len(x))

        x = np.insert(x, i, x[i - 1])
        y = np.insert(y, i, y[i - 1])
    return x, y


def add_noise_to_file(file_name, output_base, g2p=None, repeats=50, dup_prob=0.2):
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
    # if "val" in output_base:
    #     all_phonemes = random.choices(all_phonemes, k=10000)
    final_clean = []
    final_noise = []
    print("start adding noise")
    for i in tqdm(range(repeats)):
        for phonemes in all_phonemes:
            noise_phones = raplace_random(phonemes, random.random())
            if dup_prob > 0:
                phonemes, noise_phones = replace_random_with_dup(phonemes, noise_phones, random.random() * dup_prob)
            final_clean.append(" ".join([str(p) for p in phonemes]))

            final_noise.append(" ".join([str(p) for p in noise_phones]))
    dup_name = "_dup" if dup_prob > 0 else ""
    with open(output_base + f"{dup_name}_clean.txt", 'w') as file:
        file.write("\n".join(final_clean))
    with open(output_base + f"{dup_name}_noise.txt", 'w') as file:
        file.write("\n".join(final_noise))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['tm', 'lr'], default="tm")
    parser.add_argument('--add_dup', type=float, default=0.3)

    vars = parser.parse_args()
    if vars.dataset == "tm":
        train_file = "data/TIMIT_TRAIN_PH.txt"
        valid_output_file = "data/TIMIT_TRAIN_VAL_PH"
        test_file = "data/TIMIT_TEST_PH.txt"
        g2p = None
        train_repeats = 1000
        val_test_repeats = 10
    else:  # 'lr'
        train_file = "data/lr_train.txt"
        valid_output_file = "data/lr_train_val"
        test_file = "data/lr_test.txt"
        g2p = None  # G2p()
        train_repeats = 10
        val_test_repeats = 1
    add_noise_to_file(train_file, output_base=train_file.replace(".txt", ""), g2p=g2p, repeats=train_repeats,
                      dup_prob=vars.add_dup)

    add_noise_to_file(train_file, output_base=valid_output_file, g2p=g2p, repeats=val_test_repeats,
                      dup_prob=vars.add_dup)

    add_noise_to_file(test_file, output_base=test_file.replace(".txt", ""), g2p=g2p, repeats=val_test_repeats,
                      dup_prob=vars.add_dup)
