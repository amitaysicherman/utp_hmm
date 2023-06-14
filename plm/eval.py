from train import input_size, PhonemesDataset, mask_value, padding_value
from x_transformers import TransformerWrapper, Encoder
import random
import torch
from train import get_model

cp_file = "./models/change_66.cp"
data_path = 'LR960_PH.npz'
data_len_path = 'LR960_PH_LEN.txt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model()
model.load_state_dict(torch.load(cp_file, map_location=torch.device('cpu')))

model = model.to(device)
model.eval()

dataset = PhonemesDataset(data_path=data_path, data_len_path=data_len_path)

for i in range(10):
    x = dataset[i]
    y = x.clone()

    x = x.unsqueeze(0)
    x = x.to(device)
    r = random.random()
    mask = torch.zeros_like(x).float().uniform_(0, 1) <= r
    mask[x == padding_value] = False
    random_tokens = torch.randint_like(x, input_size)
    x[mask] = random_tokens[mask]
    output = model(x)

    predicted_labels = torch.argmax(output.squeeze(0), dim=1)
    predicted_labels = predicted_labels[y != padding_value]

    mask = mask[0]
    mask = mask[y != padding_value]
    y = y[y != padding_value]
    x=x[x!=padding_value]
    print("----")
    print("mask", r)
    print("len y", len(y))
    print("len mask", sum(mask.flatten()))
    correct_predictions = (predicted_labels[~mask] == y[~mask]).sum().item()
    n = len(y[~mask])
    print("no mask", correct_predictions / n)
    correct_predictions = (predicted_labels[mask] == y[mask]).sum().item()
    n = len(y[mask])
    print("acc mak", correct_predictions / len(y), len(y[mask]))

    print("acc tot", sum(predicted_labels == y), sum(x == y))
