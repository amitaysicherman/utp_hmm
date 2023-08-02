# sbatch --time=3-0 --gres=gpu:1,vmem:24g --mem=64G --wrap "python feature_extractor_superv_vad_unsuperv_p.py"

import argparse
import glob
import os
import fairseq
import numpy as np
import pandas as pd
import torch
import torchaudio
from npy_append_array import NpyAppendArray
from tqdm import tqdm
from model import NextFrameClassifier
import dataclasses
import joblib
import math

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
                     'th': 31, 'uh': 32, 'uw': 33, 'v': 34, 'w': 35, 'y': 36, 'z': 37, 'zh': 38}


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
    mapping = np.zeros((100, 39))
    for audio_file in tqdm(glob.glob(os.path.join(timit_base, "*", "*", "*WAV"))):
        phonemes_ranges = read_phonemes_range(audio_file.replace(".WAV", ".PHN"))
        clusters = hubert_features_extractor.get_cluster(audio_file)
        fill_mapping(mapping, clusters, phonemes_ranges)
    matching = mapping.argmax(axis=1)
    pd.DataFrame(mapping).to_csv("mapping.csv", index=False)
    pd.DataFrame(matching).to_csv("matching.csv", index=False)


def load_pseg_model(pseg_model):
    model = NextFrameClassifier()
    ckpt = torch.load(pseg_model, map_location="cpu")
    weights = ckpt["state_dict"]
    weights = {k.replace("NFC.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model = model.to('cpu')
    model = model.eval()
    return model


def save_timit_feaures(timit_base, output_base, hubert_cp, pseg_model, km_model, use_kmeans):
    model = load_pseg_model(pseg_model)

    hfe = HubertFeaturesExtractor(hubert_cp, km_model, use_kmeans)
    os.makedirs(output_base, exist_ok=True)
    features_file = os.path.join(output_base, "features.npy")
    if os.path.exists(features_file):
        os.remove(features_file)
    features_output_file = NpyAppendArray(features_file)
    lengthes = []
    names = []
    all_phonemes = []
    for audio_file in tqdm(glob.glob(os.path.join(timit_base, "*", "*", "*WAV"))):
        features, phonemes = hfe.extract_features(audio_file, model)
        features_output_file.append(features)
        lengthes.append(str(len(features)))
        names.append(audio_file.replace(timit_base, ""))
        all_phonemes.append(phonemes)
    with open(os.path.join(output_base, "features.length"), 'w') as f:
        f.write("\n".join(lengthes))
    with open(os.path.join(output_base, "features.names"), 'w') as f:
        f.write("\n".join(names))
    with open(os.path.join(output_base, "features.phonemes"), 'w') as f:
        f.write("\n".join(all_phonemes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timit_base', type=str, default="/cs/labs/adiyoss/amitay.sich/TIMIT/data/TRAIN/")
    parser.add_argument('--hubert_cp', type=str, default=)
    parser.add_argument('--pseg_model', type=str, default='./models/timit+_pretrained.ckpt')
    parser.add_argument('--kmeans_model', type=str, default=)
    parser.add_argument('--use_kmeans', type=int, default=1)

    parser.add_argument('--output_base', type=str, default='./data/sup_vad_km/')

    args = parser.parse_args()
    save_timit_feaures(args.timit_base, args.output_base, args.hubert_cp, args.pseg_model, args.kmeans_model,
                       args.use_kmeans)
