# sbatch --gres=gpu:4,vmem:24g --mem=75G --time=3-0 --wrap "python train_mlm.py --model=transformer --data_train=data/lr_train.txt --data_test=data/lr_test.txt --size=medium --epochs=10000"


from mlm import MLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torch.nn.parallel import DataParallel

import torch
from utils import args_parser, get_model, PADDING_VALUE, MASK_VALUE, get_config_name, Scores, save_model, N_TOKENS

MASK_PROB = 0.2
RANDOM_MAX_PROB = 1 - MASK_PROB


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
    config_name = get_config_name(args)

    train_scores = Scores("train", config_name)
    test_scores = Scores("test", config_name)
    warmup = 10
    ephochs = range(-warmup, 0) + range(args.epochs + 1)
    random_token_probs = [0] * warmup + list(np.linspace(0, RANDOM_MAX_PROB, args.epochs + 1))
    for epoch, random_token_prob in zip(ephochs, random_token_probs):
        trainer = MLM(
            model,
            mask_token_id=MASK_VALUE,  # the token id reserved for masking
            pad_token_id=PADDING_VALUE,  # the token id for padding
            mask_prob=MASK_PROB,  # masking probability for masked language modeling
            random_token_prob=random_token_prob,  # masking probability for a random token
            num_tokens=N_TOKENS,  # number of tokens in the dataset
            replace_prob=1.0 if random_token_prob > 0 else 0.9,
            # 1- probability that token will not be masked, but included in loss, as detailed in the epaper
        ).to(device)

        model.train()
        for data in tqdm(train_data):
            data = data.to(device)
            loss, logits, y = trainer(data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_scores.update(y, logits, loss.item())
        model.eval()
        with torch.no_grad():
            for data in tqdm(test_data):
                data = data.to(device)
                loss, logits, y = trainer(data)
                test_scores.update(y, logits, loss.item())
        if epoch % 50 == 0:
            save_model(model, optimizer, args, epoch, suf="_" + str(random_token_prob))
        print(f"random_token_prob {random_token_prob} Epoch", epoch)
        print(train_scores)
        print(test_scores)
        train_scores.save_and_reset()
        test_scores.save_and_reset()
