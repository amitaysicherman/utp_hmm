import glob
from mapping import TIMIT_61_39, phonemes_to_index
from tqdm import tqdm



def file_to_phonemes(file, ignore_sil=True):
    with open(file, 'r') as f:
        phonemes = [x.split()[2] for x in f.read().splitlines()]
    phonemes = [TIMIT_61_39[phoneme] for phoneme in phonemes]
    if ignore_sil:
        phonemes = [p for p in phonemes if p != "sil"]
    phonemes = [p.upper() for p in phonemes]
    phonemes_index = [str(phonemes_to_index[p]) for p in phonemes]
    phonemes=" ".join(phonemes)
    phonemes_index=" ".join(phonemes_index)
    return phonemes, phonemes_index

def get_phonemes(base_dir, ignore_sil=True):
    files = glob.glob(base_dir + "*/*/*.PHN")

    names = [x.replace(base_dir, "") for x in files]
    with open("TIMIT_"+base_dir.split("/")[-2] + "_names.txt", 'w') as f:
        for name in names:
            f.write(f'{name}\n')

    all_phonemes = []
    all_phonemes_indexes = []
    for file in tqdm(files):
        phonemes, phonemes_index = file_to_phonemes(file, ignore_sil)
        all_phonemes.append(phonemes)
        all_phonemes_indexes.append(phonemes_index)
    return all_phonemes, all_phonemes_indexes


def write_phonemes_to_file(base_dir, output_file):
    all_phonemes, all_phonemes_indexes = get_phonemes(base_dir)
    with open(output_file, 'w') as f:
        f.write("\n".join(all_phonemes))
    with open(output_file.replace(".txt", "_IDX.txt"), 'w') as f:
        f.write("\n".join(all_phonemes_indexes))

if __name__ == '__main__':
    timit_train = "/cs/labs/adiyoss/amitay.sich/TIMIT/data/TRAIN/"
    write_phonemes_to_file(timit_train, "TIMIT_TRAIN_PH.txt")
    timit_test = "/cs/labs/adiyoss/amitay.sich/TIMIT/data/TEST/"
    write_phonemes_to_file(timit_test, "TIMIT_TEST_PH.txt")
