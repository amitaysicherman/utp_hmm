import glob
from mapping import TIMIT_61_39, phonemes_to_index
from tqdm import tqdm


def get_phonemes(base_dir, ignore_sil=True):
    files = glob.glob(base_dir + "*/*/*.PHN")

    names = [x.replace(base_dir, "") for x in files]
    with open(base_dir.split("/")[-1] + "_names.txt", 'w') as f:
        for name in names:
            f.write(f'{name}\n')

    all_phonemes = []
    all_phonemes_indexes = []
    for file in files:
        with open(file, 'r') as f:
            phonemes = [x.split()[2] for x in f.read().splitlines()]
        phonemes = [TIMIT_61_39[phoneme] for phoneme in phonemes]
        if ignore_sil:
            phonemes = [p for p in phonemes if p != "sil"]
        phonemes = [p.upper() for p in phonemes]
        phonemes_index = [str(phonemes_to_index[p]) for p in phonemes]
        all_phonemes.append(" ".join(phonemes))
        all_phonemes_indexes.append(" ".join(phonemes_index))
    return all_phonemes, all_phonemes_indexes


def write_phonemes_to_file(base_dir, output_file):
    all_phonemes, all_phonemes_indexes = get_phonemes(base_dir)
    with open(output_file, 'w') as f:
        f.write("\n".join(all_phonemes))
    with open(output_file.replace(".txt", "_IDX.txt"), 'w') as f:
        f.write("\n".join(all_phonemes_indexes))


timit_train = "/cs/labs/adiyoss/amitay.sich/TIMIT/data/TRAIN/"
write_phonemes_to_file(timit_train, "TIMIT_TRAIN_PH.txt")
timit_test = "/cs/labs/adiyoss/amitay.sich/TIMIT/data/TEST/"
write_phonemes_to_file(timit_test, "TIMIT_TEST_PH.txt")
