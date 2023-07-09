# sbatch --gres=gpu:4,vmem:24g --mem=75G --time=3-0 --wrap "python train_mlm.py --model=transformer --data_train=data/lr_train.txt --data_test=data/lr_test.txt"


from mlm import MLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torch.nn.parallel import DataParallel

import torch
from utils import args_parser, get_model, PADDING_VALUE, MASK_VALUE, get_config_name, Scores, save_model


class PhonemesDataset(Dataset):
    def __init__(self, data_file, max_len, padding_value=PADDING_VALUE):
        with open(data_file, 'r') as f:
            data = f.read().splitlines()
        data = [list(map(int, line.split())) for line in data]
        self.pad_data = []
        for seq in data:
            seq_len = len(seq)
            if seq_len > max_len:
                start_index = np.random.randint(0, seq_len - max_len)
                seq = seq[start_index:start_index + max_len]
                seq = np.array(list(seq))
            else:
                seq = np.array(list(seq) + [padding_value] * (max_len - len(seq)))
            self.pad_data.append(seq)

    def __len__(self):
        return len(self.pad_data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.pad_data[idx])


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = args_parser()
    train_ds = PhonemesDataset(args.data_train, args.max_len)
    test_ds = PhonemesDataset(args.data_test, args.max_len)
    model = get_model(args.model, args.size, args.max_len, args.drop_out, MASK_VALUE + 1)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    train_data = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_data = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = MLM(
        model,
        mask_token_id=MASK_VALUE,  # the token id reserved for masking
        pad_token_id=PADDING_VALUE,  # the token id for padding
        mask_prob=0.15,  # masking probability for masked language modeling
        replace_prob=0.90,
        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
    ).to(device)
    for epoch in range(args.epochs):
        config_name = get_config_name(args)
        train_scores = Scores("train", config_name)
        test_scores = Scores("test", config_name)
        model.train()
        for data in tqdm(train_data):
            data=data.to(device)
            loss, logits, y = trainer(data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_scores.update(data, logits, loss)
        model.eval()
        with torch.no_grad():
            for data in tqdm(test_data):
                data=data.to(device)
                loss, logits, y = trainer(data)
                test_scores.update(data, logits, loss)
        save_model(model, optimizer, args, epoch)
        print("Epoch", epoch)
        print(train_scores)
        print(test_scores)

        train_scores.save_and_reset()
        test_scores.save_and_reset()
