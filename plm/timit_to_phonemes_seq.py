import glob
from mapping import TIMIT_61_39, phonemes_to_index
from tqdm import tqdm


def file_to_phonemes(file):
    with open(file, 'r') as f:
        phonemes = [x.split()[2] for x in f.read().splitlines()]
    phonemes = [TIMIT_61_39[phoneme] for phoneme in phonemes]
    phonemes = [p for p in phonemes if p != "sil"]
    phonemes = [phonemes[0]] + [p for i, p in enumerate(phonemes[1:]) if p != phonemes[i]]
    phonemes_index = [str(phonemes_to_index[p.upper()]) for p in phonemes]
    phonemes_index = " ".join(phonemes_index)
    return phonemes_index


def get_phonemes(base_dir):
    files = glob.glob(base_dir + "*/*/*.PHN")
    all_phonemes_indexes = []
    for file in tqdm(files):
        all_phonemes_indexes.append(file_to_phonemes(file))
    return all_phonemes_indexes


def write_phonemes_to_file(base_dir, output_file):
    all_phonemes_indexes = get_phonemes(base_dir)
    with open(output_file.replace(".txt", "_IDX.txt"), 'w') as f:
        f.write("\n".join(all_phonemes_indexes))

if __name__ == '__main__':
    timit_train = "/cs/labs/adiyoss/amitay.sich/TIMIT/data/TRAIN/"
    write_phonemes_to_file(timit_train, "data/TIMIT_TRAIN_PH.txt")
    timit_test = "/cs/labs/adiyoss/amitay.sich/TIMIT/data/TEST/"
    write_phonemes_to_file(timit_test, "data/TIMIT_TEST_PH.txt")
