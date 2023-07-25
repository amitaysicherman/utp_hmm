# sbatch --killable --gres=gpu:1,vmem:8g --mem=16G --time=0-3 --wrap "python eval_model_with_superv_mapping.py --cp=models/long_marix_1_20130000.cp"
import numpy as np
import torch
from tqdm import tqdm
from utils import get_model, PADDING_VALUE, N_TOKENS
from train_long_reaplce_matrix import PhonemesDataset, PROB, ONE
from mapping import phonemes_to_index
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cp', type=str, default="./models/long_marix_16090000.cp")
args = parser.parse_args()

with open("./pseg/data/p_superv/features.clusters") as f:
    code100 = f.read().splitlines()
code100 = [[int(y) for y in x.split()] for x in code100]

with open("./pseg/data/p_superv/features.phonemes") as f:
    phonemes = f.read().splitlines()
phonemes = [[phonemes_to_index[y.upper()] if y != "dx" else phonemes_to_index['T'] for y in x.split()] for x in
            phonemes]

sep = PADDING_VALUE
noise_sep = 100
data = []
code_data = []
max_len = 1024
sample = []
sample_code = []
for p, c in zip(phonemes, code100):
    sample += p
    sample_code += c
    sample += [sep]
    sample_code += [noise_sep]

    if len(sample) >= max_len:
        sample = sample[:max_len]
        sample_code = sample_code[:max_len]
        data.append(sample)
        code_data.append(sample_code)
        sample = []
        sample_code = []

data = torch.LongTensor(data)
code_data = torch.LongTensor(code_data)

units_to_phonemes = np.zeros((100, len(phonemes_to_index)))
for i, (u, p) in enumerate(tqdm(zip(sum(code100, []), sum(phonemes, [])))):
    units_to_phonemes[u, p] += 1

superv_mapping = units_to_phonemes.argmax(axis=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model("transformer", "medium", 1024, 0.0, vocab=101, output_dim=N_TOKENS + 1)
model.load_state_dict(torch.load(args.cp, map_location=torch.device('cpu')))
model = model.to(device)

train_dataset = PhonemesDataset(size=100, type_=PROB)

scores = []
for x, y in tqdm(train_dataset):
    x = x.to(device)
    res = model(x.unsqueeze(0))
    pred = res.argmax(dim=-1)
    scores.append((pred.detach().cpu().numpy()[0] == y.numpy()).mean())
print(np.mean(scores))

scores = []
model_units_to_phonemes = np.zeros((100, len(phonemes_to_index)))

for x, y in tqdm(zip(code_data, data), total=len(code_data)):
    x = x.to(device)
    res = model(x.unsqueeze(0))[0]
    for i in range(len(x)):
        c_ = x[i].item()
        if c_ == noise_sep:
            continue

        model_units_to_phonemes[c_, :] += res[i].detach().cpu().numpy()[:-1]

    pred = res.argmax(dim=-1)
    scores.append((pred.detach().cpu().numpy() == y.numpy()).mean())
model_superv_mapping = model_units_to_phonemes.argmax(axis=1)[:100]
clusters_scores = (model_superv_mapping == superv_mapping).sum()

print("scores", np.mean(scores))
print("cluster", clusters_scores)

scores = []
clusters_scores = 0
model_units_to_phonemes = np.zeros((100, len(phonemes_to_index)))

for x, y in tqdm(zip(code_data, data), total=len(code_data)):
    x_save=x[:]

    x = [model_superv_mapping[i] if i != noise_sep else noise_sep for i in x]
    x= torch.LongTensor(x)
    x = x.to(device)
    res = model(x.unsqueeze(0))[0]
    for i in range(len(x)):
        c_ = x_save[i].item()
        if c_ == noise_sep:
            continue

        model_units_to_phonemes[c_, :] += res[i].detach().cpu().numpy()[:-1]

    pred = res.argmax(dim=-1)
    scores.append((pred.detach().cpu().numpy() == y.numpy()).mean())

model_superv_mapping = model_units_to_phonemes.argmax(axis=1)[:100]
clusters_scores = (model_superv_mapping == superv_mapping).sum()


print("scores", np.mean(scores))
print("cluster", clusters_scores)
