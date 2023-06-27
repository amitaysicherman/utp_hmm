import torch
import fairseq
import soundfile as sf
from scipy.spatial.distance import cosine
import numpy as np
import matplotlib.pyplot as plt

step = 320

TIMIT_61_39 = {'aa': 'aa', 'ae': 'ae', 'ah': 'ah', 'ao': 'aa', 'aw': 'aw', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er',
               'ay': 'ay', 'b': 'b', 'bcl': 'sil', 'ch': 'ch', 'd': 'd', 'dcl': 'sil', 'dh': 'dh', 'dx': 't',
               'eh': 'eh', 'el': 'l', 'em': 'm', 'en': 'n', 'eng': 'ng', 'epi': 'sil', 'er': 'er', 'ey': 'ey', 'f': 'f',
               'g': 'g', 'gcl': 'sil', 'h#': 'sil', 'hh': 'hh', 'hv': 'hh', 'ih': 'ih', 'ix': 'ih', 'iy': 'iy',
               'jh': 'jh', 'k': 'k', 'kcl': 'sil', 'l': 'l', 'm': 'm', 'n': 'n', 'ng': 'ng', 'nx': 'n', 'ow': 'ow',
               'oy': 'oy', 'p': 'p', 'pau': 'sil', 'pcl': 'sil', 'q': 'sil', 'r': 'r', 's': 's', 'sh': 'sh', 't': 't',
               'tcl': 'sil', 'th': 'th', 'uh': 'uh', 'uw': 'uw', 'ux': 'uw', 'v': 'v', 'w': 'w', 'y': 'y', 'z': 'z',
               'zh': 'sh'}


def plot_results(times,phoenemes, w_sig, h_sig):
    times=(times/step).round()*step
    plt.figure(figsize=(20, 5))
    x = [i * step for i in range(len(w_sig))]
    plt.plot(x, w_sig,".-", label='wav2vec',c='g')
    plt.plot(x, h_sig,".-", label='hubert', c='b')
    plt.vlines(times, 0, 1, colors='r', linestyles='dashed', label='phonemes')
    for i,p in enumerate(phoenemes):
        plt.text(times[i], 0.5, p, fontsize=12, color='black', horizontalalignment='right')
    plt.legend()
    plt.show()


def features_to_signal(featues):
    return [0]+[cosine(featues[i], featues[i + 1]) for i in range(len(featues) - 1)]


def get_phonemes_from_file(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines()
    times = []
    phonemes=[]
    prev_p = ""
    for line in lines:
        _, end, p = line.split()
        p = TIMIT_61_39[p]
        if p != prev_p:
            times.append(int(end))
            phonemes.append(p)
            prev_p = p

    return np.array(times)[:-1],phonemes[:-1]


def read_audio(fname, channel_id=None):
    wav, sr = sf.read(fname)
    if channel_id is not None:
        assert wav.ndim == 2, \
            f"Expected stereo input when channel_id is given ({fname})"
        assert channel_id in [1, 2], \
            "channel_id is expected to be in [1, 2]"
        wav = wav[:, channel_id - 1]
    if wav.ndim == 2:
        wav = wav.mean(-1)
    return wav


class HubertFeaturesExtractor:
    def __init__(self, checkpoint_path="./models/hubert_base_ls960.pt", layer=6, use_cuda=False):
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
        self.model = models[0]
        if use_cuda:
            self.model.cuda()
        self.model.eval()
        self.layer = layer
        self.use_cuda = use_cuda

    def get_feats(self, file_path, channel_id=None):
        x = read_audio(file_path, channel_id)
        with torch.no_grad():
            source = torch.from_numpy(x).view(1, -1).float()
            if self.use_cuda:
                source = source.cuda()
            features = self.model.extract_features(
                source=source,
                padding_mask=None,
                mask=False,
                output_layer=self.layer,
            )[0]
            return features[0].detach().cpu().numpy()


class Wav2VecFeatureReader:
    """
    Wrapper class to run inference on Wav2Vec 2.0 model.
    Helps extract features for a given audio file.
    """

    def __init__(self, checkpoint_path="/home/amitay/PycharmProjects/utp_hmm/plm/pseg/models/wav2vec.pt", layer=14,
                 use_cuda=False):
        state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(
            checkpoint_path
        )
        w2v_args = state["args"]
        self.task = fairseq.tasks.setup_task(w2v_args)
        model = self.task.build_model(w2v_args)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        self.model = model
        self.layer = layer
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()

    def get_feats(self, file_path, channel_id=None):
        x = read_audio(file_path, channel_id)
        with torch.no_grad():
            source = torch.from_numpy(x).view(1, -1).float()
            if self.use_cuda:
                source = source.cuda()
            res = self.model(
                source=source, mask=False, features_only=True, layer=self.layer
            )
            return res["layer_results"][self.layer][0].squeeze(1).detach().cpu().numpy()




if __name__=="__main__":

    wfe = Wav2VecFeatureReader()
    hfe = HubertFeaturesExtractor()

    file_name = "/home/amitay/PycharmProjects/utp_hmm/plm/pseg/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV"
    p_file = file_name.replace("WAV", "PHN")
    times,phonemes = get_phonemes_from_file(p_file)
    w2v_feats = wfe.get_feats(file_name)
    w2v_sig = features_to_signal(w2v_feats)

    hubert_feats = hfe.get_feats(file_name)
    hubert_sig = features_to_signal(hubert_feats)
    plot_results(times,phonemes, w2v_sig, hubert_sig)