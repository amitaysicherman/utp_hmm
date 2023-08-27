# sbatch --gres=gpu:1,vmem:24g --mem=75G -c5 --time=7-0 --wrap "python cluster_to_phonemes_bart.py"
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from tqdm import tqdm
import os
from scipy.special import softmax
from scipy.spatial.distance import cdist
from transformers import BartConfig, BartForConditionalGeneration
from dataclasses import dataclass
from jiwer import wer
import argparse

BATCH_SIZE = 1  # 32
LR = 1e-4
log_steps = 500
save_update_steps = 10_000
gen_steps = 50_000

warmup_steps = 100
last_config = False
parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, default="lr")

args = parser.parse_args()

ds = args.ds
if ds == "lr":
    phonemes_file = "data/lr_train.txt"
    phonemes_file_test = "data/lr_test.txt"
    MAX_DS_SIZE = 2 ** 17
else:
    phonemes_file = "data/TIMIT_NS_TRAIN_PH_IDX.txt"
    phonemes_file_test = "data/TIMIT_NS_TEST_PH_IDX.txt"
    MAX_DS_SIZE = 4000
load_cp = ""
config_name = f"learn_mapping_bart_{ds}"
gen_file = f"results/{config_name}_gen.txt"
EPOCHS = 1_000
test_size = 1_000
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_file = f"results/{config_name}.txt"


@dataclass
class Scores:
    train_loss = 0.0
    train_acc = 0.0
    train_count = 0
    test_loss = 0.0
    test_acc = 0.0
    test_count = 0

    def resset_train(self):
        self.train_loss = 0.0
        self.train_acc = 0.0
        self.train_count = 0.0

    def resset_test(self):
        self.test_loss = 0.0
        self.test_acc = 0.0
        self.test_count = 0

    def reset(self):
        self.resset_train()
        self.resset_test()

    def update_value(self, loss, acc, train_test):

        if train_test == "train":
            self.train_loss += loss
            self.train_acc += acc
            self.train_count += 1
        else:
            self.test_loss += loss
            self.test_acc += acc
            self.test_count += 1

    def update_values_from_output(self, outputs, y, train_test):
        loss = outputs.loss.item()
        preds = outputs.logits.argmax(dim=-1)
        mask = y != PAD_TOKEN
        acc = ((preds[mask] == y[mask]).sum() / y[mask].numel()).item()
        self.update_value(loss, acc, train_test)

    def train_to_str(self):
        train_loss = self.train_loss / self.train_count if self.train_count > 0 else 0
        train_acc = self.train_acc / self.train_count if self.train_count > 0 else 0
        return f"TRAIN:{train_loss},{train_acc}"

    def test_to_str(self):
        test_loss = self.test_loss / self.test_count if self.test_count > 0 else 0
        test_acc = self.test_acc / self.test_count if self.test_count > 0 else 0
        return f"TEST:{test_loss},{test_acc}"

    def to_file(self, train_test):
        if train_test == "train":
            with open(output_file, "a") as f:
                f.write(self.train_to_str() + "\n")
        else:
            with open(output_file, "a") as f:
                f.write(self.test_to_str() + "\n")


def random_gaussian(n, dim=2):
    point = np.random.normal(size=(n, dim))
    point /= np.linalg.norm(point, axis=1, keepdims=True)
    return point


class PhonemesDataset(Dataset):
    def __init__(self, phonemes_file, type_, dup, max_len=MAX_LENGTH, samples_count=train_dataset_size, size=-1):
        with open(phonemes_file, 'r') as f:
            phonemes_data = f.readlines()
        self.phonemes_data = [[int(x) for x in line.strip().split()] for line in phonemes_data]
        self.size = size
        self.max_len = max_len
        self.type = type_
        self.dup = dup
        self.samples_count = samples_count
        self.data = []

        self.build_data()

    def build_data(self):
        max_line_index = len(self.phonemes_data) if self.size == -1 else self.size
        max_line_index = min(max_line_index, len(self.phonemes_data))
        self.data = []
        for _ in range(self.samples_count):
            sample = [START_TOKEN]
            while len(sample) < self.max_len:
                sample += self.phonemes_data[np.random.randint(0, max_line_index)]
                sample += [SEP]
            sample = sample[:self.max_len - 1] + [END_TOKEN]
            self.data.append(sample)

    def update_config(self, type_, dup, size):
        self.type = type_
        self.dup = dup
        self.size = size
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

        final_clean = [clean[0]]  # start token
        final_noise = [clean[0]]  # start token
        range_units = np.arange(N_CLUSTERS)
        finish = False
        for c in clean[1:]:
            if finish:
                break
            if c == SEP:
                final_clean.append(SEP)
                final_noise.append(SEP)
                if len(final_noise) == self.max_len - 1:
                    final_clean.append(END_TOKEN)
                    final_noise.append(END_TOKEN)
                    break

            elif c == END_TOKEN:
                final_clean.append(END_TOKEN)
                final_noise.append(END_TOKEN)
                break
            else:
                final_clean.append(c)
                for _ in range(length.pop()):

                    if self.type == ONE:
                        new_token = random.choice(inv_mapping[c])
                    else:  # self.type == SPHERE:
                        new_token = np.random.choice(range_units, p=inv_mapping[c])

                    new_token = CLUSTERS_FIRST_TOKEN + new_token
                    final_noise.append(new_token)
                    if len(final_noise) == self.max_len - 1:
                        final_clean.append(END_TOKEN)
                        final_noise.append(END_TOKEN)
                        finish = True
                        break

        final_clean += [PAD_TOKEN] * (MAX_LENGTH - len(final_clean))
        final_noise += [PAD_TOKEN] * (MAX_LENGTH - len(final_noise))

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
                                    size=curr_size)
    train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataset = PhonemesDataset(phonemes_file_test, type_=curr_type, dup=curr_dup,
                                   size=-1, samples_count=test_size)
    test_data = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return train_dataset, train_data, test_dataset, test_data


def save(model, optimizer, i, best_score, update_best, conf_type, conf_dup, conf_size):
    data_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': i,
        'best_score': best_score,
        'conf_type': conf_type,
        'conf_dup': conf_dup,
        'conf_size': conf_size
    }
    torch.save(data_save, f"models/{config_name}_last.cp")
    if update_best:
        torch.save(data_save, f"models/{config_name}_best.cp")


def load_last(model, optimizer):
    if not os.path.exists(f"models/{config_name}_last.cp"):
        return 0, 0, ONE, False, 2
    checkpoint = torch.load(f"models/{config_name}_last.cp", map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    load_step = checkpoint['step']
    best_score = checkpoint['best_score']
    conf_type = checkpoint['conf_type']
    conf_dup = checkpoint['conf_dup']
    conf_size = checkpoint['conf_size']
    return load_step, best_score, conf_type, conf_dup, conf_size


def gen(model: BartForConditionalGeneration, dataset, split_name, i):
    with open(gen_file, "a") as f:
        f.write("-----------------------------")
        f.write(f"split: {split_name}\n")
        f.write("-----------------------------")
    wer_gen_scores = []
    for j, (x_gen, y_ref) in enumerate(dataset):

        x_gen = x_gen.to(device)

        min_new_tokens = int(0.25 * MAX_LENGTH)

        y_gen = model.generate(x_gen, max_new_tokens=MAX_LENGTH, min_new_tokens=min_new_tokens,
                               top_k=6, num_beams=100, output_scores=True, return_dict_in_generate=True,
                               num_return_sequences=4)['sequences'][0]

        y_gen = y_gen.cpu().numpy().tolist()
        y_gen = " ".join([str(x) for x in y_gen if x not in [PAD_TOKEN, START_TOKEN, END_TOKEN, SEP]])

        y_ref = y_ref[0].cpu().numpy().tolist()
        y_ref = " ".join([str(x) for x in y_ref if x not in [PAD_TOKEN, START_TOKEN, END_TOKEN, SEP]])
        with open(gen_file, "a") as f:
            f.write(
                f'x: {" ".join([str(x) for x in x_gen[0].cpu().numpy().tolist() if x not in [PAD_TOKEN, START_TOKEN, END_TOKEN, SEP]])}\n')
            f.write(f"gen: {y_gen}\n")
            f.write(f"ref: {y_ref}\n\n")
        wer_gen_scores.append(wer(y_ref, y_gen))
        if j > 5:
            break
    print(f"step {i},split {split_name}, wer: {np.mean(wer_gen_scores)}")


# main:
if __name__ == '__main__':

    model = get_model()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    i, best_test_acc, curr_type, curr_dup, curr_size = load_last(model, optimizer)
    print(
        f"load cp-  i:{i}, best_test_acc:{best_test_acc}, curr_type:{curr_type}, curr_dup:{curr_dup}, curr_size:{curr_size}")
    model = model.train()

    scores = Scores()
    for epoch in range(EPOCHS):
        train_dataset, train_data, test_dataset, test_data = get_datasets()
        pbar = tqdm(train_data, total=len(train_data))

        for x_train, y_train in pbar:
            i += 1
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            outputs = model(input_ids=x_train, labels=y_train, output_hidden_states=True)
            scores.update_values_from_output(outputs, y_train, "train")
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % save_update_steps == 0:
                model.eval()
                with torch.no_grad():
                    for x_test, y_test in test_data:
                        x_test = x_test.to(device)
                        y_test = y_test.to(device)
                        outputs = model(input_ids=x_test, labels=y_test, output_hidden_states=True)
                        scores.update_values_from_output(outputs, y_train, "test")
                model.train()
                update_best = False
                if scores.test_acc > best_test_acc:
                    best_test_acc = scores.test_acc
                    update_best = True
                scores.to_file("test")
                scores.resset_test()
                save(model, optimizer, i, best_test_acc, update_best, curr_type, curr_dup, curr_size)

            # if i % gen_steps == 0:
            #     model.eval()
            #     gen(model, train_data, "train", i)
            #     gen(model, test_data, "test", i)
            #     model.train()

            if i % warmup_steps == 0:
                scores.to_file("train")

                if not last_config:
                    curr_type, curr_dup, curr_size, is_update = step_config(curr_type, curr_dup, curr_size,
                                                                            scores.train_acc / scores.train_count)
                    if curr_type == SPHERE and curr_dup and curr_size == MAX_DS_SIZE:
                        last_config = True
                    if is_update:
                        with open(output_file, "a") as f:
                            f.write(f"step {i}, update config to {curr_type}, {curr_dup}, {curr_size}" + "\n")
                        config_update_counter = 0
                        break
                scores.resset_train()
