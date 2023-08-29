# sbatch --time=0-12 --gres=gpu:1,vmem:24g --mem=32G -c4 --wrap "python lr_train_to_idx.py"

from nltk.corpus import cmudict
from mapping import phonemes_to_index
import glob
import os
import fairseq
import torch
import torchaudio
from tqdm import tqdm
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HubertFeaturesExtractor:
    def __init__(self, ckpt_path, km_path, layer=6):
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = models[0].to(device)
        self.layer = layer
        self.km = joblib.load(km_path)

    def extract_features(self, audio_file):
        audio, _ = torchaudio.load(audio_file)
        audio = audio.to(device)
        features = self.model.extract_features(
            source=audio,
            padding_mask=None,
            mask=False,
            output_layer=self.layer,
        )[0][0].detach().cpu().numpy()
        clusters = self.km.predict(features)
        return clusters


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

hubert_cp = "./models/hubert_base_ls960.pt"
km_model = "./models/km100.bin"
hfe = HubertFeaturesExtractor(hubert_cp, km_model)

files = glob.glob('/cs/dataset/Download/adiyoss/librispeech/LibriSpeech/train-*/*/*/*.trans.txt')
files = sorted(files)

all_phonemes = []
all_cluseters = []
all_names = []
skip_count = 0
tot_count = 0
pbar = tqdm(files)
for file in pbar:
    dir_name = os.path.dirname(file)
    pbar.set_description(f"[{skip_count}/{tot_count}]")

    with open(file, 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        tot_count += 1
        suf_name, text = line.split(' ', 1)
        phonemes = text_to_phonemes(text)
        if phonemes is None:
            skip_count += 1
            continue
        all_phonemes.append(phonemes)
        file_name = os.path.join(dir_name, suf_name + '.flac')
        new_clusters = hfe.extract_features(file_name)
        new_clusters = [new_clusters[0]] + [new_clusters[i] for i in range(1, len(new_clusters)) if
                                            new_clusters[i] != new_clusters[i - 1]]
        new_clusters = [str(x) for x in new_clusters]
        new_clusters = " ".join(new_clusters)
        all_cluseters.append(new_clusters)
        all_names.append(file_name)

with open("data/lr_train_idx.txt", 'w') as f:
    f.write("\n".join(all_phonemes))

with open("data/lr_train_clusters.txt", 'w') as f:
    f.write("\n".join(all_cluseters))

with open("data/lr_train_names.txt", 'w') as f:
    f.write("\n".join(all_names))
