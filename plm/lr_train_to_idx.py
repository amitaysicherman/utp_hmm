import glob

from nltk.corpus import cmudict
from mapping import phonemes_to_index
from tqdm import tqdm


def text_to_phonemes(text):
    words = text.lower().split()
    phonemes = []
    for word in words:
        # Lookup the word in CMU dict
        if word in pronouncing_dict:
            # Take the first pronunciation option
            word_phonemes = pronouncing_dict[word][0]
            phonemes.extend(word_phonemes)
        else:
            return None

    phonemes = [p[:-1] if p[-1].isnumeric() else p for p in phonemes]
    phonemes = [phonemes_to_index[p] for p in phonemes]
    phonemes = " ".join([str(p) for p in phonemes])
    return phonemes


pronouncing_dict = cmudict.dict()

files = glob.glob('/cs/dataset/Download/adiyoss/librispeech/LibriSpeech/train-*/*/*/*.trans.txt')
files = sorted(files)

all_phonemes = []
skip_count = 0
tot_count = 0
pbar = tqdm(files)

for file in pbar:
    pbar.set_description(f"[{skip_count}/{tot_count}]")

    with open(file, 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        tot_count += 1
        _, text = line.split(' ', 1)
        phonemes = text_to_phonemes(text)
        if phonemes is None:
            skip_count += 1
            continue
        all_phonemes.append(phonemes)
with open("data/lr_train_idx.txt", 'w') as f:
    f.write("\n".join(all_phonemes))
