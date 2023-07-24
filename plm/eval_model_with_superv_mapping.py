import numpy as np
import torch
from tqdm import tqdm
from utils import get_model, PADDING_VALUE, N_TOKENS
from train_long_reaplce_matrix import PhonemesDataset,PROB,ONE
from mapping import phonemes_to_index

# cp_file = "./models/prep_random_small_timit_15.cp"

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

# acc = 0
# tot = 0
# for x, y in zip(code_data.numpy(), data.numpy()):
#     y_hat = [superv_mapping[x_] for x_ in x]
#     print((y_hat == y).mean())
#     acc += (y_hat == y).sum()
#     tot += len(y)

model = get_model("transformer", "medium", 1024, 0.0, vocab=101, output_dim=N_TOKENS + 1)
model.load_state_dict(torch.load("./models/long_marix_16090000.cp", map_location=torch.device('cpu')))
train_dataset = PhonemesDataset(size=100,type_=PROB)
# for x,y in train_dataset:
model_units_to_phonemes = np.zeros((100 + 1, len(phonemes_to_index) + 1))

# for x, y in zip(code_data, data):
for x,y in train_dataset:

    res = model(x.unsqueeze(0))
    pred = res.argmax(dim=-1)
    for c_, l_ in zip(x, pred[0]):
        model_units_to_phonemes[c_.item(), l_.item()] += 1
    print('s', (pred.detach().numpy()[0] == y.numpy()).mean())
    model_superv_mapping = model_units_to_phonemes.argmax(axis=1)[:100]
    print('m', (model_superv_mapping == superv_mapping).sum())
