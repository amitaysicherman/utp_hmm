from typing import Tuple
from mapping import phonemes_to_index
import torch.nn as nn
import torch


class ConvEncoder(nn.Module):
    def __init__(
            self,
            model_dim,
            kernel_size,
            n_layers,
            vocab_size: int = len(phonemes_to_index) + 1,
            dropout: float = 0.0,
            conv_bias: bool = True,
            fc_bias: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.emb = nn.Embedding(vocab_size, model_dim)

        self.conv_layers = nn.ModuleList()
        for i in range(n_layers):
            conv = nn.Sequential(
                nn.Conv1d(model_dim, model_dim, kernel_size, bias=conv_bias, padding="same"),
                nn.Dropout(p=dropout),
                nn.GELU())
            self.conv_layers.append(conv)
        self.fc = nn.Linear(model_dim, vocab_size, bias=fc_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        x = x.transpose(1, 2)
        for conv in self.conv_layers:
            x = conv(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        return x


def get_conv_encoder_model(size="small"):
    if size == "small":
        return ConvEncoder(model_dim=384, kernel_size=6, n_layers=3)
    if size == "medium":
        return ConvEncoder(model_dim=768, kernel_size=9, n_layers=6)
    if size == "large":
        return ConvEncoder(model_dim=1024, kernel_size=12, n_layers=9)
    raise ValueError(f"Unknown size {size}")


if __name__ == '__main__':
    m = get_conv_encoder_model()
    print(sum(p.numel() for p in m.parameters() if p.requires_grad))

    x = torch.randint(0, len(phonemes_to_index), (2, 100))
    y = m(x)

    m = get_conv_encoder_model("medium")
    print(sum(p.numel() for p in m.parameters() if p.requires_grad))

    m = get_conv_encoder_model("large")
    print(sum(p.numel() for p in m.parameters() if p.requires_grad))
