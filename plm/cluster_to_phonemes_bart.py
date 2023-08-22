# sbatch --gres=gpu:1,vmem:24g --mem=75G -c5 --time=7-0 --wrap "python cluster_to_phonemes_bart.py"
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from tqdm import tqdm

from scipy.special import softmax
from scipy.spatial.distance import cdist
from transformers import BartConfig, BartForConditionalGeneration
from dataclasses import dataclass

BATCH_SIZE = 1  # 32
LR = 1e-4
log_steps = 500
save_update_step = 10_000

phonemes_file = "data/lr_train.txt"
phonemes_file_test = "data/lr_test.txt"
MAX_DS_SIZE = 2 ** 17

load_cp = ""
config_name = "learn_mapping_bart"

EPOCHS = 100
test_size = 100
train_dataset_size = 50_000

ONE = 0
SPHERE = 2
MAX_LENGTH = 512
PHONEMES_LAST_TOKEN = 38
CLUSTERS_FIRST_TOKEN = PHONEMES_LAST_TOKEN + 1
N_CLUSTERS = 100
CLUSTERS_LAST_TOKEN = CLUSTERS_FIRST_TOKEN + N_CLUSTERS
PAD_TOKEN = CLUSTERS_LAST_TOKEN + 1
SEP = PAD_TOKEN + 1
START_TOKEN = SEP + 1
END_TOKEN = START_TOKEN + 1
N_TOKENS = END_TOKEN + 1


@dataclass
class Scores:
    loss: float = 0
    acc: float = 0
    test_loss: float = 0
    test_acc: float = 0
    loss_list = None
    acc_list = None
    test_loss_list = None
    test_acc_list = None
    output_file = f"results/{config_name}.txt"

    def mean_value(self, name, list_name):
        if getattr(self, list_name) is not None:
            setattr(self, name, sum(getattr(self, list_name)) / len(getattr(self, list_name)))
            setattr(self, list_name, None)

    def mean_train(self):
        self.mean_value("loss", "loss_list")
        self.mean_value("acc", "acc_list")

    def mean_test(self):
        self.mean_value("test_loss", "test_loss_list")
        self.mean_value("test_acc", "test_acc_list")

    def mean_all(self):
        self.mean_train()
        self.mean_test()

    def update_value(self, name, score):
        if getattr(self, name) is None:
            setattr(self, name, [score])
        else:
            getattr(self, name).append(score)

    def to_str(self):
        return f"{self.loss},{self.acc},{self.test_loss},{self.test_acc}"

    def to_file(self):
        with open(self.output_file, "a") as f:
            f.write(self.to_str() + "\n")


def random_gaussian(n, dim=2):
    point = np.random.normal(size=(n, dim))
    point /= np.linalg.norm(point, axis=1, keepdims=True)
    return point


class PhonemesDataset(Dataset):
    def __init__(self, phonemes_file, type_, dup, max_len=MAX_LENGTH, size=train_dataset_size, phonemes_lines_count=-1):
        with open(phonemes_file, 'r') as f:
            phonemes_data = f.readlines()
        self.phonemes_data = [[int(x) for x in line.strip().split()] for line in phonemes_data]
        self.phonemes_lines_count = phonemes_lines_count
        self.max_len = max_len
        self.type = type_
        self.dup = dup
        self.size = size
        self.build_data()

    def build_data(self):
        max_line_index = len(self.phonemes_data) if self.phonemes_lines_count == -1 else self.phonemes_lines_count
        self.data = []
        for _ in range(self.size):
            sample = [START_TOKEN]
            while len(sample) < self.max_len:
                sample += self.phonemes_data[np.random.randint(0, max_line_index)]
                sample += [SEP]
            sample = sample[:self.max_len - 1] + [END_TOKEN]
            self.data.append(sample)

    def update_data(self, phonemes_lines_count=-1):
        self.phonemes_lines_count = phonemes_lines_count
        self.build_data()

    def __len__(self):
        return len(self.data)

    def build_mapping_one(self):
        units_mapping = list(range(PHONEMES_LAST_TOKEN + 1))
        units_mapping += [random.randint(0, PHONEMES_LAST_TOKEN) for _ in
                          range(N_CLUSTERS - (PHONEMES_LAST_TOKEN + 1))]
        random.shuffle(units_mapping)
        units_mapping = np.array(units_mapping)
        inv_mapping = {i: [] for i in range(PHONEMES_LAST_TOKEN + 1)}
        for i, u in enumerate(units_mapping):
            inv_mapping[u].append(i)
        return inv_mapping

    def build_mapping_sphere(self):
        phonemes = random_gaussian(PHONEMES_LAST_TOKEN + 1)
        clusters = random_gaussian(N_CLUSTERS)
        cosine_distances = 100 * (1 - cdist(phonemes, clusters, metric='cosine'))
        probabilities = softmax(cosine_distances, axis=0)
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        np.random.shuffle(probabilities)
        return probabilities

    def build_mapping(self):
        if self.type == ONE:
            return self.build_mapping_one()
        elif self.type == SPHERE:
            return self.build_mapping_sphere()
        else:
            raise ValueError("Unknown type")

    def add_noise(self, clean):
        inv_mapping = self.build_mapping()

        values = np.arange(5)
        random_numbers = np.random.random(4)
        sorted_numbers = np.sort(np.concatenate(([0, 1], random_numbers)))
        weights = np.diff(sorted_numbers)
        np.random.shuffle(weights)
        if self.dup:
            length = random.choices(values, weights=weights, k=len(clean))

        else:
            length = [1] * len(clean)

        final_clean = []
        final_noise = []
        range_units = np.arange(N_CLUSTERS)
        for c in clean:
            if len(final_noise) > self.max_len - 1:
                break

            if c in [START_TOKEN, END_TOKEN, PAD_TOKEN, SEP]:
                final_clean.append(c)
                final_noise.append(c)
            else:
                final_clean.append(c)
                for _ in range(length.pop()):
                    if self.type == ONE:
                        new_token = random.choice(inv_mapping[c])
                    else:  # self.type == SPHERE:
                        new_token = np.random.choice(range_units, p=inv_mapping[c])

                    new_token = CLUSTERS_FIRST_TOKEN + new_token
                    final_noise.append(new_token)

        if len(final_clean) < MAX_LENGTH:
            final_clean += [PAD_TOKEN] * (MAX_LENGTH - len(final_clean))
        if len(final_noise) < MAX_LENGTH:
            final_noise += [PAD_TOKEN] * (MAX_LENGTH - len(final_noise))

        if len(final_noise) > MAX_LENGTH:
            final_noise = final_noise[:MAX_LENGTH - 1] + [END_TOKEN]
        if len(final_clean) > MAX_LENGTH:
            final_clean = final_clean[:MAX_LENGTH - 1] + [END_TOKEN]

        return final_clean, final_noise

    def __getitem__(self, idx):
        clean, noise = self.add_noise(self.data[idx])
        clean = torch.LongTensor(clean)
        noise = torch.LongTensor(noise)
        return noise, clean


def step_config(cur_type, cur_dup, curr_size, score):
    is_update = False
    if score > 0.6:
        if curr_size < MAX_DS_SIZE:
            curr_size *= 2
            is_update = True
        elif cur_type == ONE:
            cur_type = SPHERE
            is_update = True
        elif cur_type == SPHERE and not cur_dup:
            cur_dup = True
            is_update = True
        print(f"update config {curr_size}, {cur_type} {cur_dup}\n", flush=True)
    return cur_type, cur_dup, curr_size, is_update


def get_model() -> BartForConditionalGeneration:
    d_model = 768
    nhead = 12
    num_layers = 12

    config = BartConfig(vocab_size=N_TOKENS + 1, max_position_embeddings=MAX_LENGTH, encoder_layers=num_layers,
                        encoder_ffn_dim=d_model,
                        encoder_attention_heads=nhead, decoder_layers=num_layers, decoder_ffn_dim=d_model,
                        decoder_attention_heads=nhead,
                        d_model=d_model, pad_token_id=PAD_TOKEN, bos_token_id=START_TOKEN, eos_token_id=END_TOKEN,
                        decoder_start_token_id=START_TOKEN, forced_eos_token_id=END_TOKEN)  # Set vocab size
    model = BartForConditionalGeneration(config)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'{params:,} trainable parameters')
    return model


def get_datasets():
    train_dataset = PhonemesDataset(phonemes_file, type_=curr_type, dup=curr_dup,
                                    phonemes_lines_count=curr_size)
    train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataset = PhonemesDataset(phonemes_file_test, type_=curr_type, dup=curr_dup,
                                   phonemes_lines_count=curr_size, size=test_size)
    test_data = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return train_dataset, train_data, test_dataset, test_data


def save(model, optimizer, best_score=None):
    torch.save(model.state_dict(), f"models/{config_name}_last.cp")
    torch.save(optimizer.state_dict(), f"models/{config_name}_last_opt.cp")
    if best_score:
        torch.save(model.state_dict(), f"models/{config_name}_best.cp")
        torch.save(optimizer.state_dict(), f"models/{config_name}_best_opt.cp")


# main:
if __name__ == '__main__':
    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    load_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model = model.train()
    curr_type = ONE
    curr_dup = False
    curr_size = 2

    scores = Scores()

    best_test_acc = 0

    train_dataset, train_data, test_dataset, test_data = get_datasets()
    for epoch in range(EPOCHS):
        pbar = tqdm(train_data, total=len(train_data))
        for i, (x_train, y_train) in enumerate(pbar):
            pbar.set_description(scores.to_str())
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            outputs = model(input_ids=x_train, labels=y_train, output_hidden_states=True)

            scores.update_value("loss_list", outputs.loss.item())

            preds = outputs.logits.argmax(dim=-1)
            scores.update_value("acc_list", ((preds == y_train).sum() / y_train.numel()).item())

            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % log_steps == 0:
                scores.mean_train()
                scores.to_file()
                curr_type, curr_dup, curr_size, is_update = step_config(curr_type, curr_dup, curr_size, scores.acc)
                if is_update:
                    train_dataset.update_data(curr_size)
                    train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

            if i % save_update_step == 0:
                model.eval()
                with torch.no_grad():
                    for x_test, y_test in test_data:
                        x_test = x_test.to(device)
                        y_test = y_test.to(device)
                        outputs = model(input_ids=x_test, labels=y_test, output_hidden_states=True)
                        preds = outputs.logits.argmax(dim=-1)
                        scores.update_value("test_loss_list", outputs.loss.item())
                        scores.update_value("test_acc_list", ((preds == y_test).sum() / y_test.numel()).item())

                model.train()
                scores.mean_test()
                scores.to_file()
                if scores.test_acc > best_test_acc:
                    best_test_acc = scores.test_acc
                    save(model, optimizer, best_test_acc)
                else:
                    save(model, optimizer, None)
