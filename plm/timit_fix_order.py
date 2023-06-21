import glob
from timit_to_phonemes_seq import file_to_phonemes
from tqdm import tqdm


BASE_DIR="/cs/labs/adiyoss/amitay.sich/TIMIT/data/TRAIN/"
files = glob.glob(BASE_DIR + "*/*/*.PHN")

name_to_phonemes={}
for file_name in tqdm(files):
    name=file_name.replace(BASE_DIR, "").replace(".PHN", "")
    _, phonemes_indexes = file_to_phonemes(file_name)
    name_to_phonemes[name]=phonemes_indexes


with open("/cs/labs/adiyoss/amitay.sich/textless-speech-disorders/datasets/TM_hubert_100_hifi/units.txt") as f:
    lines=[eval(line) for line in f.read().splitlines()]
name_to_units={}
for line in tqdm(lines):
    name_to_units[line["audio"].split(".")[0]]=line["units"]

codes=[]
phonemnes=[]
for name in name_to_units:
    codes.append(name_to_units[name])
    phonemnes.append(name_to_phonemes[name])


with open("/cs/labs/adiyoss/amitay.sich/utp_hmm/data/TIMIT_UPDATE_PH.txt", 'w') as f:
    for phonemes in phonemnes:
        f.write(f'{phonemes}\n')
with open("/cs/labs/adiyoss/amitay.sich/utp_hmm/data/TIMIT_UPDATE_CODES.txt", 'w') as f:
    for code in codes:
        f.write(f'{code}\n')