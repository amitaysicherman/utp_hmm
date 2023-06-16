import random

import pandas as pd
from train import PhonemesDataset, padding_value
import torch
from tqdm import tqdm
from train import get_model
import torch.nn.functional as F
from scipy.stats import entropy

cp_file = "./models/prep_random_small_46.cp"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model()
model.load_state_dict(torch.load(cp_file, map_location=torch.device('cpu')))

model = model.to(device)
model.eval()


def eval_dataset(dataset, tot=10_000):
    losses = []
    type = []
    ps = []
    ids = []
    tot = min(tot, len(dataset))
    probs = []
    ent_softmax = []
    for _ in tqdm(range(tot)):
        i = random.randint(0, len(dataset) - 1)
        x, y = dataset[i]
        x = x.to(device)
        y = y.to(device)

        logits = model(x.unsqueeze(0))

        loss = F.cross_entropy(
            logits.transpose(1, 2),
            y.unsqueeze(0),
            ignore_index=padding_value,
            reduction="none"
        )[0]

        probs_ = F.softmax(logits, dim=-1)[0]
        probs_ = probs_.cpu().numpy()
        y = y.cpu().numpy()
        loss = loss.detach().cpu().numpy()

        ent_softmax_ = entropy(probs_, axis=-1)
        ent_softmax_ = ent_softmax_[y != padding_value]
        ent_softmax.extend(ent_softmax_.tolist())

        probs_ = probs_[range(len(probs_)), y]
        probs_ = probs_[y != padding_value]
        probs.extend(probs_.tolist())

        loss = loss[y != padding_value]
        x = x[y != padding_value]
        y = y[y != padding_value]
        losses.extend(loss.tolist())
        type.extend((x != y).astype(int).tolist())
        p = (x != y).astype(int).mean()
        p = round(p, 2)
        ps.extend([p] * len(loss))
        ids.extend([i] * len(loss))

    return pd.DataFrame({"id": ids, "loss": losses, "type": type, "p": ps, "probs": probs, "ent_softmax": ent_softmax})


with torch.no_grad():
    train_data = PhonemesDataset("LR960")
    eval_dataset(train_data).to_csv("train_losses.csv")

    test_data = PhonemesDataset("LRTEST")

    eval_dataset(test_data).to_csv("test_losses.csv")
