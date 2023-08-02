# sbatch --killable --gres=gpu:1,vmem:8g --mem=16G --time=0-3 --wrap "python eval_model_with_superv_mapping.py --cp=models/long_marix_1_20130000.cp"
import numpy as np
import torch
from tqdm import tqdm
from utils import get_model, PADDING_VALUE, N_TOKENS
from train_long_reaplce_matrix import PhonemesDataset, PROB, ONE
from mapping import phonemes_to_index
import argparse
from jiwer import wer

sep = PADDING_VALUE
noise_sep = 100
max_len = 1024
superv_seg = False


def data_to_tensor(phonemes, code100):
    data = []
    code_data = []
    sample = []
    sample_code = []
    for p, c in zip(phonemes, code100):
        sample += p
        sample_code += c
        sample += [sep]
        sample_code += [noise_sep]
        if len(sample_code) >= max_len:
            if len(sample) >= max_len:
                sample = sample[:max_len]
            else:
                sample += [sep] * (max_len - len(sample))
            sample_code = sample_code[:max_len]
            data.append(sample)
            code_data.append(sample_code)
            sample = []
            sample_code = []
    data = torch.LongTensor(data)
    code_data = torch.LongTensor(code_data)
    return data, code_data


def eval_data_with_model(data, code_data):
    scores = []
    wer_score = []
    model_units_to_phonemes = np.zeros((100, len(phonemes_to_index)))
    for x, y in tqdm(zip(code_data, data), total=len(code_data)):
        y = y.numpy().flatten()
        x = x.to(device)
        res = model(x.unsqueeze(0))[0]
        for i in range(len(x)):
            c_ = x[i].item()
            if c_ == noise_sep:
                continue
            model_units_to_phonemes[c_, :] += res[i].detach().cpu().numpy()[:-1]
        pred = res.argmax(dim=-1)
        y_hat = pred.detach().cpu().numpy()
        y_hat[x.cpu().numpy() == noise_sep] = sep
        y_hat = remove_sep_and_dup(y_hat)
        y = remove_sep_and_dup(y)
        print(y.shape, y_hat.shape)
        if len(y) == len(y_hat):
            scores.append((y == y_hat).mean())

        wer_score.append(wer_np(y, y_hat))
    return np.mean(scores), np.mean(wer_score), model_units_to_phonemes


def evel_data_with_mapping(phonemes, code100, mapping):
    scores_wer = []
    for x, y in tqdm(zip(code100, phonemes), total=len(phonemes)):
        pred = np.array([mapping[i] if i != noise_sep else noise_sep for i in x])
        pred = remove_sep_and_dup(pred)
        y = np.array(y)
        scores_wer.append(wer_np(y, pred))
    return np.mean(scores_wer)


def remove_sep_and_dup(x):
    x = np.array([x[0]] + [x[i] for i in range(1, len(x)) if x[i] != x[i - 1]])
    # x = x[x != sep]
    return x


def wer_np(y, y_hat):
    y = " ".join([str(x) for x in y])
    y_hat = " ".join([str(x) for x in y_hat])
    return wer(y, y_hat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cp', type=str, default="./models/long_marix_1True_27130000.cp")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("transformer", "medium", 1024, 0.0, vocab=101, output_dim=N_TOKENS + 1)
    model.load_state_dict(torch.load(args.cp, map_location=torch.device('cpu')))
    model = model.to(device)
    ###############################################
    # dataset score:
    ###############################################
    train_dataset = PhonemesDataset(size=100, type_=PROB)

    scores = []
    wer_scores = []
    for x, y in tqdm(train_dataset):
        x = x.to(device)
        res = model(x.unsqueeze(0))
        pred = res.argmax(dim=-1)
        y = y.numpy()
        y_hat = pred.detach().cpu().numpy()[0]
        scores.append((y == y_hat).mean())
        wer_scores.append(wer_np(y, y_hat))
    print("scores dataset", np.mean(scores), "WER dataset", np.mean(wer_scores))

    ###############################################
    # Build Supervision mapping:
    ###############################################
    cluster_file = "./pseg/data/p_superv/features.clusters"
    with open(cluster_file) as f:
        code100 = f.read().splitlines()
    code100 = [[int(y) for y in x.split()] for x in code100]
    with open("./pseg/data/p_superv/features.phonemes") as f:
        phonemes = f.read().splitlines()
    phonemes = [[phonemes_to_index[y.upper()] if y != "dx" else phonemes_to_index['T'] for y in x.split()] for x in
                phonemes]
    units_to_phonemes = np.zeros((100, len(phonemes_to_index)))
    for i, (u, p) in enumerate(tqdm(zip(sum(code100, []), sum(phonemes, [])))):
        units_to_phonemes[u, p] += 1
    superv_mapping = units_to_phonemes.argmax(axis=1)

    scores_wer = evel_data_with_mapping(phonemes, code100, superv_mapping)
    print("Supervision Clustering Score WER: ", scores_wer)

    ###############################################
    # Learn Mapping from model:
    ###############################################

    with open("./pseg/data/sup_vad/features.clusters") as f:
        code100_dup = f.read().splitlines()
    code100_dup = [[int(y) for y in x.split()] for x in code100_dup]
    code100 = []
    for line in code100_dup:
        code100.append([line[0]] + [line[i] for i in range(1, len(line)) if line[i] != line[i - 1]])
    with open("./pseg/data/sup_vad/features.phonemes") as f:
        phonemes = f.read().splitlines()
    phonemes = [[phonemes_to_index[y.upper()] if y != "dx" else phonemes_to_index['T'] for y in x.split()] for x in
                phonemes]

    data, code_data = data_to_tensor(phonemes, code100)
    wer_sep = evel_data_with_mapping(phonemes, code100, superv_mapping)
    print("Supervision Real Data Clustering Score WER: ", wer_sep)

    scores, wer, model_units_to_phonemes = eval_data_with_model(data, code_data)
    model_superv_mapping = model_units_to_phonemes.argmax(axis=1)[:100]
    wer_model = evel_data_with_mapping(phonemes, code100, model_superv_mapping)
    print("Supervision Model Clustering Score WER: ", wer_model)

    print("Clusters Eq Superv (Argmax)", (model_superv_mapping == superv_mapping).sum())
    print("Clusters Eq Superv (TOT %)", (np.abs(model_units_to_phonemes - units_to_phonemes)).mean())

    ###############################################
    # Learned Mapping + Model
    ###############################################
    scores = []
    scores_wer = []
    model_units_to_phonemes = np.zeros((100, len(phonemes_to_index)))
    for x, y in tqdm(zip(code_data, data), total=len(code_data)):
        x_save = x[:]
        x = [model_superv_mapping[i] if i != noise_sep else noise_sep for i in x]
        x = torch.LongTensor(x)
        x = x.to(device)
        res = model(x.unsqueeze(0))[0]
        for i in range(len(x)):
            c_ = x_save[i].item()
        if c_ == noise_sep:
            continue
        model_units_to_phonemes[c_, :] += res[i].detach().cpu().numpy()[:-1]
        pred = res.argmax(dim=-1)
        y_hat = pred.detach().cpu().numpy()
        y_hat[x.cpu().numpy() == noise_sep] = sep

        y_hat = remove_sep_and_dup(y_hat)
        y = y.numpy()
        y = remove_sep_and_dup(y)
        if len(y) == len(y_hat):
            scores.append((y == y_hat).mean())
        scores_wer.append(wer_np(y, y_hat))
    print("Mapping + Model Cluster To Phoneme scores: ", np.mean(scores))
    print("Mapping + Model Cluster To Phoneme WER: ", np.mean(scores_wer))
    model_superv_mapping = model_units_to_phonemes.argmax(axis=1)[:100]
    print("Clusters Eq Superv (Argmax)", (model_superv_mapping == superv_mapping).sum())
    print("Clusters Eq Superv (TOT %)", (np.abs(model_units_to_phonemes - units_to_phonemes)).mean())
