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
            x[i] = np.random.randint(0, 39)
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="/home/amitay/PycharmProjects/utp_hmm/data/phonemes.txt")
    parser.add_argument('--output_prefix', type=str, default="TMTR")
    parser.add_argument('--do_phonemes', type=bool, default=False)
    vars = parser.parse_args()
    g2p = G2p()
    with open(vars.input_file, 'r') as file:
        lines = file.read().splitlines()

    all_phonemes = []
    all_length = []
    for line in tqdm(lines):
        line = line.strip().upper()
        if vars.do_phonemes:
            sentence = " ".join(line.split(' ')[1:])  # Extract the sentence after the tab character
            phonemes = g2p(sentence)
        else:
            phonemes = line.split(' ')
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
