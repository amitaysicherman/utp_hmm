import numpy as np
from g2p_en import G2p
from tqdm import tqdm
import argparse

phonemes_to_index = {' ': 0, 'AA': 1, 'AE': 2, 'AH': 3, 'AO': 4, 'AW': 5, 'AY': 6, 'B': 7, 'CH': 8, 'D': 9, 'DH': 10,
                     'EH': 11, 'ER': 12, 'EY': 13, 'F': 14, 'G': 15, 'HH': 16, 'IH': 17, 'IY': 18, 'JH': 19, 'K': 20,
                     'L': 21, 'M': 22, 'N': 23, 'NG': 24, 'OW': 25, 'OY': 26, 'P': 27, 'R': 28, 'S': 29, 'SH': 30,
                     'T': 31, 'TH': 32, 'UH': 33, 'UW': 34, 'V': 35, 'W': 36, 'Y': 37, 'Z': 38, 'ZH': 39}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="libri_test.txt")
    parser.add_argument('--output_prefix', type=str, default="LRTEST")
    parser.add_argument('--do_phonemes', type=bool, default=True)
    parser.add_argument('--space_p', type=float, default=0.2)

    vars=parser.parse_args()

    # Create an instance of the G2p class
    g2p = G2p()
    # Read the text file
    with open(vars.input_file, 'r') as file:
        lines = file.read().splitlines()


    all_phonemes = []
    all_length = []
    for line in tqdm(lines):
        if vars.do_phonemes:
            line = line.strip().upper()
            sentence = " ".join(line.split(' ')[1:])  # Extract the sentence after the tab character
            phonemes = g2p(sentence)
            phonemes = [p[:-1] if p[-1].isnumeric() else p for p in phonemes]
            phonemes = [p for p in phonemes if p != "'"]
        else:
            phonemes = line.upper().split()
            phonemes = [p if p !="DX" else "D" for p in phonemes]

        phonemes = [p for p in phonemes if p != " "]
        s_to_add = int(len(phonemes) * vars.space_p)
        for i in range(s_to_add):
            phonemes.insert(np.random.randint(0, len(phonemes)), " ")

        all_length.append(len(phonemes))

        for p in phonemes:
            if p not in phonemes_to_index:
                print(p)
        all_phonemes.extend([phonemes_to_index[p] for p in phonemes])

    all_phonemes_index = np.array(all_phonemes, dtype=np.int8)
    tot_len = sum([int(l) for l in all_length])
    assert tot_len == len(all_phonemes_index)
    np.savez_compressed(f"{vars.output_prefix}_PH",a=all_phonemes_index)
    with open(f'{vars.output_prefix}_PH_LEN.txt', 'w') as file:
        file.write("\n".join([str(l) for l in all_length]))
