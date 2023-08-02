# sbatch --time=1-0 --gres=gpu:1,vmem:24g --mem=64G --wrap "python timit_superv_mapping.py"

import glob
import os
import fairseq
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
import dataclasses
import joblib

step = 320
SIL = "sil"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMIT_61_39 = {'aa': 'aa', 'ae': 'ae', 'ah': 'ah', 'ao': 'aa', 'aw': 'aw', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er',
               'ay': 'ay', 'b': 'b', 'bcl': 'sil', 'ch': 'ch', 'd': 'd', 'dcl': 'sil', 'dh': 'dh', 'dx': 't',
               'eh': 'eh', 'el': 'l', 'em': 'm', 'en': 'n', 'eng': 'ng', 'epi': 'sil', 'er': 'er', 'ey': 'ey', 'f': 'f',
               'g': 'g', 'gcl': 'sil', 'h#': 'sil', 'hh': 'hh', 'hv': 'hh', 'ih': 'ih', 'ix': 'ih', 'iy': 'iy',
               'jh': 'jh', 'k': 'k', 'kcl': 'sil', 'l': 'l', 'm': 'm', 'n': 'n', 'ng': 'ng', 'nx': 'n', 'ow': 'ow',
               'oy': 'oy', 'p': 'p', 'pau': 'sil', 'pcl': 'sil', 'q': 'sil', 'r': 'r', 's': 's', 'sh': 'sh', 't': 't',
               'tcl': 'sil', 'th': 'th', 'uh': 'uh', 'uw': 'uw', 'ux': 'uw', 'v': 'v', 'w': 'w', 'y': 'y', 'z': 'z',
               'zh': 'sh'}
phonemes_to_index = {'aa': 0, 'ae': 1, 'ah': 2, 'ao': 3, 'aw': 4, 'ay': 5, 'b': 6, 'ch': 7, 'd': 8, 'dh': 9, 'eh': 10,
                     'er': 11, 'ey': 12, 'f': 13, 'g': 14, 'hh': 15, 'ih': 16, 'iy': 17, 'jh': 18, 'k': 19, 'l': 20,
                     'm': 21, 'n': 22, 'ng': 23, 'ow': 24, 'oy': 25, 'p': 26, 'r': 27, 's': 28, 'sh': 29, 't': 30,
                     'th': 31, 'uh': 32, 'uw': 33, 'v': 34, 'w': 35, 'y': 36, 'z': 37, 'zh': 38, 'sil': 39}


@dataclasses.dataclass
class TimitRow:
    phonemes: int
    start: float
    end: float


def read_phonemes_range(phonemes_file):
    with open(phonemes_file) as f:
        ranges_phonemes = f.read().splitlines()
    ranges_phonemes = [p.split() for p in ranges_phonemes]
    results = []
    for p, s, e in ranges_phonemes:
        if p not in TIMIT_61_39:
            print(f"phoneme {p} not in TIMIT_61_39")
            continue
        results.append(TimitRow(phonemes=phonemes_to_index[TIMIT_61_39[p]], start=int(s) / step, end=int(e) / step))
    return results


class HubertFeaturesExtractor:
    def __init__(self, ):
        models, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(["./models/hubert_base_ls960.pt"])
        model = models[0].eval()
        self.model = model.to(device)
        self.km = joblib.load('./models/km100.bin')
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def get_cluster(self, audio_file):
        audio, _ = torchaudio.load(audio_file)
        audio = audio.to(device)
        features = self.model.extract_features(
            source=audio,
            padding_mask=None,
            mask=False,
            output_layer=6,
        )[0][0].detach().cpu().numpy()
        return self.km.predict(features)


def fill_mapping(mapping, clusters, phonemes_ranges):
    for i, cluster in enumerate(clusters):
        for pr in phonemes_ranges:
            start, end, phoneme = pr.start, pr.end, pr.phonemes
            if i <= start < i + 1:
                if i <= end < i + 1:
                    weight = end - start
                else:
                    weight = i + 1 - start
                mapping[cluster, phoneme] += weight
            elif i <= end < i + 1:
                weight = end - i
                mapping[cluster, phoneme] += weight


if __name__ == "__main__":
    timit_base = "/cs/labs/adiyoss/amitay.sich/TIMIT/data/TRAIN/"
    hubert_features_extractor = HubertFeaturesExtractor()
    mapping = np.zeros((100, len(phonemes_to_index)))
    for audio_file in tqdm(glob.glob(os.path.join(timit_base, "*", "*", "*WAV"))):
        phonemes_ranges = read_phonemes_range(audio_file.replace(".WAV", ".PHN"))
        clusters = hubert_features_extractor.get_cluster(audio_file)
        fill_mapping(mapping, clusters, phonemes_ranges)
    matching = mapping.argmax(axis=1)
    pd.DataFrame(mapping).to_csv("mapping.csv", index=False)
    pd.DataFrame(matching).to_csv("matching.csv", index=False)
