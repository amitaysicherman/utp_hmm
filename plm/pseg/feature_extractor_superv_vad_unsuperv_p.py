# sbatch --time=3-0 --gres=gpu:1,vmem:24g --mem=64G --wrap "python feature_extractor_superv_vad_unsuperv_p.py"

import argparse
import glob
import os
import fairseq
import numpy as np
import torch
import torchaudio
from npy_append_array import NpyAppendArray
from tqdm import tqdm
from model import NextFrameClassifier
from scipy.signal import find_peaks
import math

step = 320
e_index = 1
p_index = 2
SIL = "sil"
peak_to_step = 2  # 2 peaks per step (320ms vs 160ms)
prominence = 0.05
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMIT_61_39 = {'aa': 'aa', 'ae': 'ae', 'ah': 'ah', 'ao': 'aa', 'aw': 'aw', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er',
               'ay': 'ay', 'b': 'b', 'bcl': 'sil', 'ch': 'ch', 'd': 'd', 'dcl': 'sil', 'dh': 'dh', 'dx': 't',
               'eh': 'eh', 'el': 'l', 'em': 'm', 'en': 'n', 'eng': 'ng', 'epi': 'sil', 'er': 'er', 'ey': 'ey', 'f': 'f',
               'g': 'g', 'gcl': 'sil', 'h#': 'sil', 'hh': 'hh', 'hv': 'hh', 'ih': 'ih', 'ix': 'ih', 'iy': 'iy',
               'jh': 'jh', 'k': 'k', 'kcl': 'sil', 'l': 'l', 'm': 'm', 'n': 'n', 'ng': 'ng', 'nx': 'n', 'ow': 'ow',
               'oy': 'oy', 'p': 'p', 'pau': 'sil', 'pcl': 'sil', 'q': 'sil', 'r': 'r', 's': 's', 'sh': 'sh', 't': 't',
               'tcl': 'sil', 'th': 'th', 'uh': 'uh', 'uw': 'uw', 'ux': 'uw', 'v': 'v', 'w': 'w', 'y': 'y', 'z': 'z',
               'zh': 'sh'}


def get_phonemes_ranges(pseg_model, audio):
    preds = pseg_model(audio)

    preds = preds[1][0]
    preds = torch.cat([preds.index_select(dim=1, index=torch.LongTensor([0] * 1).to(preds.device)), preds], dim=1)
    preds -= preds.min(-1, keepdim=True)[0]
    preds /= preds.max(-1, keepdim=True)[0]
    preds = 1 - preds
    preds = preds[0]
    preds = preds.cpu().detach().numpy()
    xmin, xmax = preds.min(), preds.max()
    preds = (preds - xmin) / (xmax - xmin)

    peaks, _ = find_peaks(preds, prominence=prominence)
    if len(peaks) == 0:
        peaks = np.array([len(preds) - 1])

    return [(math.floor(peaks[i - 1]) / peak_to_step, math.ceil(peaks[i]) / peak_to_step) for i in range(1, len(peaks))]


def read_phonemes(phonemes_file):
    with open(phonemes_file) as f:
        ranges_phonemes = f.read().splitlines()

    ranges_phonemes = [p.split() for p in ranges_phonemes]
    ranges_phonemes = [[int(p[0]), int(p[1]), TIMIT_61_39[p[2]]] for p in ranges_phonemes]

    phonemes = [TIMIT_61_39[p[2]] for p in ranges_phonemes]
    phonemes = [p for p in phonemes if p != SIL]
    phonemes = [phonemes[0]] + [p for i, p in enumerate(phonemes[1:]) if p != phonemes[i - 1]]
    phonemes = " ".join(phonemes)

    vad_ranges = [(x, y) for (x, y, z) in ranges_phonemes if z != SIL]

    return phonemes, vad_ranges


class HubertFeaturesExtractor:
    def __init__(self, ckpt_path, layer=6):
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = models[0].to(device)
        self.layer = layer
        self.step = 320

    def extract_features(self, audio_file, pseg_model):
        audio, _ = torchaudio.load(audio_file)
        phonemes, vad_ranges = read_phonemes(audio_file.replace(".WAV", ".PHN"))
        audio = torch.cat([audio[:, start:end] for start, end in vad_ranges], dim=1).to(device)
        features = self.model.extract_features(
            source=audio,
            padding_mask=None,
            mask=False,
            output_layer=self.layer,
        )[0][0].detach().cpu().numpy()

        combine_ranges = get_phonemes_ranges(pseg_model, audio.to("cpu"))
        combine_features = np.stack([features[s:e + 1].mean(axis=0) for s, e in combine_ranges])
        return combine_features, phonemes


def load_pseg_model(pseg_model):
    model = NextFrameClassifier()
    ckpt = torch.load(pseg_model, map_location="cpu")
    weights = ckpt["state_dict"]
    weights = {k.replace("NFC.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model = model.to('cpu')
    model = model.eval()
    return model


def save_timit_feaures(timit_base, output_base, hubert_cp, pseg_model):
    model = load_pseg_model(pseg_model)
    hfe = HubertFeaturesExtractor(hubert_cp)
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
    parser.add_argument('--output_base', type=str, default='./data/sup_vad/')
    parser.add_argument('--hubert_cp', type=str, default="./models/hubert_base_ls960.pt")
    parser.add_argument('--pseg_model', type=str, default='./models/timit+_pretrained.ckpt')
    args = parser.parse_args()
    save_timit_feaures(args.timit_base, args.output_base, args.hubert_cp, args.pseg_model)
