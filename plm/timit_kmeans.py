# sbatch --time=0-12 --gres=gpu:1,vmem:24g --mem=32G -c4 --wrap "python timit_kmeans.py"
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
        features = self.model.extract_features(
            source=audio,
            padding_mask=None,
            mask=False,
            output_layer=self.layer,
        )[0][0].detach().cpu().numpy()
        clusters = self.km.predict(features)
        return clusters


def save_timit_feaures(timit_base, output_file, hubert_cp, km_model):
    hfe = HubertFeaturesExtractor(hubert_cp, km_model)
    files = glob.glob(timit_base + "*/*/*.PHN")
    files = sorted(files)
    files = [x for x in files if "SA" not in x.split("/")[-1]]
    files = [x.replace(".PHN", ".WAV") for x in files]
    clusters = []
    for audio_file in tqdm(files):
        new_clusters = hfe.extract_features(audio_file)
        new_clusters = [str(x) for x in new_clusters]
        new_clusters = " ".join(new_clusters)
        clusters.append(new_clusters)
    with open(output_file, 'w') as f:
        f.write("\n".join(clusters))


if __name__ == "__main__":
    hubert_cp = "./models/hubert_base_ls960.pt"
    km_model = "./models/km100.bin"

    timit_train = "/cs/labs/adiyoss/amitay.sich/TIMIT/data/TRAIN/"
    output_train = "data/TIMIT_NS_TRAIN_CLUSTERS.txt"

    timit_test = "/cs/labs/adiyoss/amitay.sich/TIMIT/data/TEST/"
    output_test = "data/TIMIT_NS_TEST_CLUSTERS.txt"

    save_timit_feaures(timit_train, output_train, hubert_cp, km_model)
    save_timit_feaures(timit_test, output_test, hubert_cp, km_model)
