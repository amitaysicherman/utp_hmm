# sbatch --time=3-0 --gres=gpu:1,vmem:24g --mem=64G --wrap "python timit_feature_extractor.py --timit_base=/cs/labs/adiyoss/amitay.sich/TIMIT/data/TRAIN --hubert_cp=/cs/labs/adiyoss/amitay.sich/textless-speech-disorders/hubert_base_ls960.pt"

import argparse
import glob
import math
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMIT_61_39 = {'aa': 'aa', 'ae': 'ae', 'ah': 'ah', 'ao': 'aa', 'aw': 'aw', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er',
               'ay': 'ay', 'b': 'b', 'bcl': 'sil', 'ch': 'ch', 'd': 'd', 'dcl': 'sil', 'dh': 'dh', 'dx': 't',
               'eh': 'eh', 'el': 'l', 'em': 'm', 'en': 'n', 'eng': 'ng', 'epi': 'sil', 'er': 'er', 'ey': 'ey', 'f': 'f',
               'g': 'g', 'gcl': 'sil', 'h#': 'sil', 'hh': 'hh', 'hv': 'hh', 'ih': 'ih', 'ix': 'ih', 'iy': 'iy',
               'jh': 'jh', 'k': 'k', 'kcl': 'sil', 'l': 'l', 'm': 'm', 'n': 'n', 'ng': 'ng', 'nx': 'n', 'ow': 'ow',
               'oy': 'oy', 'p': 'p', 'pau': 'sil', 'pcl': 'sil', 'q': 'sil', 'r': 'r', 's': 's', 'sh': 'sh', 't': 't',
               'tcl': 'sil', 'th': 'th', 'uh': 'uh', 'uw': 'uw', 'ux': 'uw', 'v': 'v', 'w': 'w', 'y': 'y', 'z': 'z',
               'zh': 'sh'}


def replicate_first_k_frames(x, k, dim):
    return torch.cat([x.index_select(dim=dim, index=torch.LongTensor([0] * k).to(x.device)), x], dim=dim)


def max_min_norm(x):
    x -= x.min(-1, keepdim=True)[0]
    x /= x.max(-1, keepdim=True)[0]
    return x


def detect_peaks(x, lengths, prominence=0.1, width=None, distance=None):
    """detect peaks of next_frame_classifier

    Arguments:
        x {Tensor} -- batch of confidence per time
    """
    out = []

    for xi, li in zip(x, lengths):
        if type(xi) == torch.Tensor:
            xi = xi.cpu().detach().numpy()
        xi = xi[:li]  # shorten to actual length
        xmin, xmax = xi.min(), xi.max()
        xi = (xi - xmin) / (xmax - xmin)
        peaks, _ = find_peaks(xi, prominence=prominence, width=width, distance=distance)

        if len(peaks) == 0:
            peaks = np.array([len(xi) - 1])

        out.append(peaks)

    return out


def get_phonemes_ranges(pseg_model, audio):
    peak_to_step = 2  # 2 peaks per step (320ms vs 160ms)
    preds = pseg_model(audio)
    preds = preds[1][0]
    preds = replicate_first_k_frames(preds, k=1, dim=1)
    preds = 1 - max_min_norm(preds)
    preds = detect_peaks(x=preds, lengths=[preds.shape[1]], prominence=0.03)

    preds = preds[0]
    start_end = []
    for i in range(1, len(preds)):
        start = math.floor(preds[i - 1] / peak_to_step)
        end = math.ceil(preds[i] / peak_to_step)
        start_end.append((start, end))
    return start_end


def read_phonemes(phonemes_file):
    with open(phonemes_file) as f:
        lines = f.read().splitlines()
    phonemes = []
    for line in lines:
        line = line.split()
        phonemes.append(TIMIT_61_39[line[2]])
    return " ".join(phonemes)


class HubertFeaturesExtractor:
    def __init__(self, ckpt_path, layer=6):
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = models[0].to(device)
        self.layer = layer
        self.step = 320

    def extract_features(self, audio_file, pseg_model):
        audio, _ = torchaudio.load(audio_file)
        audio = audio.to(device)
        features = self.model.extract_features(
            source=audio,
            padding_mask=None,
            mask=False,
            output_layer=self.layer,
        )[0]
        features = features[0].detach().cpu().numpy()
        combine_ranges = get_phonemes_ranges(pseg_model, audio.to("cpu"))
        phonemes = read_phonemes(audio_file.replace(".wav", ".PHN"))

        combine_features = []
        for s, e in combine_ranges:
            combine_features.append(features[s:e + 1].mean(axis=0))
        return np.stack(combine_features), phonemes


def save_timit_feaures(timit_base, output_base, hubert_cp, pseg_model):
    model = NextFrameClassifier()
    ckpt = torch.load(pseg_model, map_location="cpu")
    weights = ckpt["state_dict"]
    weights = {k.replace("NFC.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.to(device)

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
    parser.add_argument('--timit_base', type=str, default="/cs/labs/adiyoss/amitay.sich/TIMIT/data/TRAIN")
    parser.add_argument('--output_base', type=str, default='./data')
    parser.add_argument('--hubert_cp', type=str,
                        default="./models/hubert_base_ls960.pt")
    parser.add_argument('--pseg_model', type=str, default='./models/timit+_pretrained.ckpt')

    args = parser.parse_args()
    save_timit_feaures(args.timit_base, args.output_base, args.hubert_cp, args.pseg_model)
