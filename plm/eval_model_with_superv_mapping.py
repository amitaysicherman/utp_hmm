import numpy as np
import torch
from tqdm import tqdm
from train import get_model

from mapping import phonemes_to_index

cp_file = "./models/timit_15.cp"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model()
model.load_state_dict(torch.load(cp_file, map_location=torch.device('cpu')))

model = model.to(device)
model.eval()

with open("../data/code100.txt") as f:
    code100 = f.read().splitlines()
code100 = [[int(y) for y in x.split()] for x in code100]

with open("../data/phonemes.txt") as f:
    phonemes = f.read().splitlines()
phonemes = [[phonemes_to_index[y.upper()] if y != "dx" else phonemes_to_index['T'] for y in x.split()] for x in
            phonemes]

units_to_phonemes = np.zeros((100, len(phonemes_to_index)))
for i, (u, p) in enumerate(tqdm(zip(sum(code100, []), sum(phonemes, [])))):
    units_to_phonemes[u, p] += 1

superv_mapping = units_to_phonemes.argmax(axis=1)

superv_phonmes = []
acc = []
acc_m = []
for c, p in zip(code100, phonemes):
    if len(p) > 50:
        continue
    p_s = [superv_mapping[u] for u in c]
    acc.append((np.array(p_s) == np.array(p)).mean())
    superv_phonmes.append(p_s)

    x = torch.LongTensor(p_s).to(device)
    logits = model(x.unsqueeze(0))
    predicted_labels = torch.argmax(logits, dim=-1)
    acc_m.append((predicted_labels[0].cpu().numpy() == np.array(p)).mean())
    print(acc[-1], acc_m[-1])

print(np.mean(acc))
print(np.quantile(acc, 0.25))
print(np.quantile(acc, 0.5))
print(np.quantile(acc, 0.75))

print(np.mean(acc_m))
print(np.quantile(acc_m, 0.25))
print(np.quantile(acc_m, 0.5))
print(np.quantile(acc_m, 0.75))
