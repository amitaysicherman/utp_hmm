import pandas as pd
from train import PhonemesDataset, padding_value
import torch
from tqdm import tqdm
from train import get_model
import torch.nn.functional as F



cp_file = "./models/prep_random_3.cp"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model()
model.load_state_dict(torch.load(cp_file, map_location=torch.device('cpu')))

model = model.to(device)
model.eval()


def eval_dataset(dataset):
    losses = []
    type = []
    ps = []
    ids = []

    for i, (x, y) in tqdm(enumerate(dataset)):
        x = x.to(device)
        y = y.to(device)

        logits = model(x.unsqueeze(0))
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            y.unsqueeze(0),
            ignore_index=padding_value,
            reduction="none"
        )[0]
        loss = loss[y != padding_value]
        x = x[y != padding_value]
        y = y[y != padding_value]
        losses.extend(loss.detach().cpu().numpy().tolist())
        type.extend((x != y).cpu().numpy().astype(int).tolist())
        p = (x != y).cpu().numpy().astype(int).mean()
        p = round(p, 1)
        ps.extend([p] * len(loss))
        ids.extend([i] * len(loss))

    return pd.DataFrame({"id": ids, "loss": losses, "type": type, "p": ps}).to_csv("losses.csv")

with torch.no_grad():
    train_data = PhonemesDataset("LR960")
    eval_dataset(train_data).to_csv("train_losses.csv")

    test_data = PhonemesDataset("LRTEST")
    eval_dataset(test_data).to_csv("test_losses.csv")
