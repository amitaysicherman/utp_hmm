import argparse

import numpy as np
import torch
from x_transformers import TransformerWrapper, Encoder
from plm.conv_encoder import get_conv_encoder_model
from mapping import phonemes_to_index

INPUT_SIZE = len(phonemes_to_index) + 1
PADDING_VALUE = INPUT_SIZE
MASK_VALUE = PADDING_VALUE + 1


def get_model(arc, size, max_len, dropout, vocab=INPUT_SIZE):
    if arc == "transformer":
        if size == "small":
            d_model = 256
            nhead = 4
            num_layers = 6
        else:
            d_model = 768
            nhead = 12
            num_layers = 12
        return TransformerWrapper(
            num_tokens=vocab + 1,
            max_seq_len=max_len,
            emb_dropout=dropout,
            attn_layers=Encoder(
                dim=d_model,
                depth=num_layers,
                heads=nhead,
                layer_dropout=dropout,
                attn_dropout=dropout,
                ff_dropout=dropout
            )
        )
    else:
        return get_conv_encoder_model(size)


def get_model_from_args(args):
    return get_model(args.model, args.size, args.max_len, args.drop_out)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=["conv", "transformer"], default="conv")
    parser.add_argument('--max_len', type=int, default=32)
    parser.add_argument('--size', type=str, default="small", choices=["small", "medium", "large"])
    parser.add_argument('--data_train', type=str, default="TIMIT_TRAIN_PH_dup")
    parser.add_argument('--data_val', type=str, default="TIMIT_TRAIN_VAL_PH_dup")
    parser.add_argument('--data_test', type=str, default="TIMIT_TEST_PH_dup")
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--drop_out', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    return args


def get_config_name(args):
    data_name = args.data_train.split("/")[-1]
    data_name = data_name.replace("_train", "")
    data_name = data_name.replace("_TRAIN", "")
    return f"{args.model}_{args.size}_{data_name}_{args.epochs}_{args.lr}_{args.drop_out}"


def save_model(model, optimizer, args, ephoc):
    config_name = get_config_name(args) + "_" + str(ephoc)
    cp_name = f"models/{config_name}.cp"
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), cp_name)
    else:
        torch.save(model.state_dict(), cp_name)
    torch.save(optimizer.state_dict(), cp_name.replace(".cp", "_opt.cp"))


class Scores:
    def __init__(self, name, output_file):
        self.name = name
        self.output_file = f"results/{output_file}.txt"
        self.loss = []
        self.acc = []

    def update(self, y, logits, loss):
        predicted_labels = torch.argmax(logits, dim=-1)
        predicted_labels = predicted_labels[y != PADDING_VALUE]
        y = y[y != PADDING_VALUE]
        acc = (predicted_labels == y).sum().item() / y.numel()
        self.loss.append(loss)
        self.acc.append(acc)

    def __repr__(self):
        return f"{self.name} loss: {np.mean(self.loss)} \n{self.name} acc: {np.mean(self.acc)}\n"

    def save(self):
        with open(self.output_file, "w") as f:
            f.write(repr(self))

    def reset(self):
        self.loss = []
        self.acc = []

    def save_and_reset(self):
        self.save()
        self.reset()
