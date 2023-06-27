import argparse
import glob
import pandas as pd
import numpy as np
import torch
import torchaudio
from model import NextFrameClassifier
from scipy.signal import find_peaks
from tqdm import tqdm

tolerance = 320

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


def detect_peaks(x, lengths, find_peaks_args):
    out = []

    for xi, li in zip(x, lengths):
        if type(xi) == torch.Tensor:
            xi = xi.cpu().detach().numpy()
        xi = xi[:li]  # shorten to actual length
        xmin, xmax = xi.min(), xi.max()
        xi = (xi - xmin) / (xmax - xmin)
        peaks, _ = find_peaks(xi, **find_peaks_args)

        if len(peaks) == 0:
            peaks = np.array([len(xi) - 1])

        out.append(peaks)

    return out


def get_metrics(precision_counter, recall_counter, pred_counter, gt_counter):
    EPS = 1e-7

    precision = precision_counter / (pred_counter + EPS)
    recall = recall_counter / (gt_counter + EPS)
    f1 = 2 * (precision * recall) / (precision + recall + EPS)

    os = recall / (precision + EPS) - 1
    r1 = np.sqrt((1 - recall) ** 2 + os ** 2)
    r2 = (-os + recall - 1) / (np.sqrt(2))
    rval = 1 - (np.abs(r1) + np.abs(r2)) / 2

    return precision, recall, f1, rval


def get_phonemes_ranges(pseg_model, audio, find_peaks_args):
    preds = pseg_model(audio)
    preds = preds[1][0]
    preds = replicate_first_k_frames(preds, k=1, dim=1)
    preds = 1 - max_min_norm(preds)
    signal = preds[0].detach().cpu().numpy().copy()
    preds = detect_peaks(x=preds, lengths=[preds.shape[1]], find_peaks_args=find_peaks_args)
    preds = preds[0] * 160
    return preds, signal


def read_phonemes(phonemes_file):
    with open(phonemes_file) as f:
        lines = f.read().splitlines()
    phonemes = []
    times = []
    prev_p = None
    for line in lines:
        start, end, phoneme = line.split()
        phoneme = TIMIT_61_39[phoneme]
        if phoneme != prev_p:
            times.append(int(end))
            prev_p = phoneme
    return np.array(times[:-1])


def load_models(pseg_model):
    model = NextFrameClassifier()
    ckpt = torch.load(pseg_model, map_location="cpu")
    weights = ckpt["state_dict"]
    weights = {k.replace("NFC.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.to('cpu')
    model.eval()
    return model


def analyze_file(model, audio_file, find_peaks_args):
    audio, _ = torchaudio.load(audio_file)
    audio = audio.to(device)
    y = read_phonemes(audio_file.replace(".WAV", ".PHN"))
    yhat, signal = get_phonemes_ranges(model, audio.to("cpu"), find_peaks_args)
    precision_counter = 0
    recall_counter = 0
    pred_counter = 0
    gt_counter = 0

    for yhat_i in yhat:
        min_dist = np.abs(y - yhat_i).min()
        precision_counter += (min_dist <= tolerance)
    for y_i in y:
        min_dist = np.abs(yhat - y_i).min()
        recall_counter += (min_dist <= tolerance)
    pred_counter += len(yhat)
    gt_counter += len(y)
    p, r, f1, rval = get_metrics(precision_counter, recall_counter, pred_counter, gt_counter)
    return p, r, f1, rval


def main(pseg_model, timit_base, find_peaks_args):
    model = load_models(pseg_model)
    scores = []
    for audio_file in tqdm(glob.glob(f"{timit_base}/*/*/*.WAV")):
        scores.append(analyze_file(model, audio_file, find_peaks_args))
    scores = pd.DataFrame(scores, columns=["precision", "recall", "f1", "rval"]).mean()
    scores = list(find_peaks_args.values()) + list(scores.values)
    score = ",".join([str(s) for s in scores])

    with open("hprarams_score.csv", "a") as f:
        f.write(score + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timit_base', type=str,
                        default="/cs/labs/adiyoss/amitay.sich/TIMIT/data/TRAIN")  # /home/amitay/PycharmProjects/utp_hmm/plm/pseg/TIMIT/TRAIN")
    parser.add_argument('--pseg_model', type=str, default='./models/timit+_pretrained.ckpt')
    parser.add_argument('--prominence', type=float, default=0.05)
    parser.add_argument('--height', type=float, default=None)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--distance', type=float, default=None)
    parser.add_argument('--width', type=float, default=None)
    args = parser.parse_args()
    to_nan = lambda x: None if x == -1 else x

    find_peaks_args = dict(prominence=to_nan(args.prominence), height=to_nan(args.height),
                           threshold=to_nan(args.threshold),
                           distance=to_nan(args.distance), width=to_nan(args.width))
    print(find_peaks_args)
    main(args.pseg_model, args.timit_base, find_peaks_args)
    args = parser.parse_args()
