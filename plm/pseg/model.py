import torch.nn.functional as F
from collections import defaultdict
import torch
import torch.nn as nn
import dataclasses


@dataclasses.dataclass
class hp:
    cosine_coef: float = 1.0
    z_proj: int = 64
    z_proj_linear: bool = True
    z_proj_dropout: float = 0
    z_dim: int = 256
    pred_steps: int = 1
    pred_offset: int = 0
    batch_shuffle: bool = False
    latent_dim: int = 0
    n_negatives: int = 1



class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


class NextFrameClassifier(nn.Module):
    def __init__(self, hp=hp()):
        super(NextFrameClassifier, self).__init__()
        self.hp = hp

        Z_DIM = hp.z_dim
        LS = hp.latent_dim if hp.latent_dim != 0 else Z_DIM

        self.enc = nn.Sequential(
            nn.Conv1d(1, LS, kernel_size=10, stride=5, padding=0, bias=False),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(LS, LS, kernel_size=8, stride=4, padding=0, bias=False),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(LS, LS, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(LS, LS, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(LS, Z_DIM, kernel_size=4, stride=2, padding=0, bias=False),
            LambdaLayer(lambda x: x.transpose(1, 2)),
        )

        if self.hp.z_proj != 0:
            if self.hp.z_proj_linear:
                self.enc.add_module(
                    "z_proj",
                    nn.Sequential(
                        nn.Dropout2d(self.hp.z_proj_dropout),
                        nn.Linear(Z_DIM, self.hp.z_proj),
                    )
                )
            else:
                self.enc.add_module(
                    "z_proj",
                    nn.Sequential(
                        nn.Dropout2d(self.hp.z_proj_dropout),
                        nn.Linear(Z_DIM, Z_DIM), nn.LeakyReLU(),
                        nn.Dropout2d(self.hp.z_proj_dropout),
                        nn.Linear(Z_DIM, self.hp.z_proj),
                    )
                )

        # # similarity estimation projections
        self.pred_steps = list(range(1 + self.hp.pred_offset, 1 + self.hp.pred_offset + self.hp.pred_steps))
        print(f"prediction steps: {self.pred_steps}")

    def score(self, f, b):
        return F.cosine_similarity(f, b, dim=-1) * self.hp.cosine_coef

    def forward(self, spect):
        device = spect.device

        # wav => latent z
        z = self.enc(spect.unsqueeze(1))

        preds = defaultdict(list)
        for i, t in enumerate(self.pred_steps):  # predict for steps 1...t
            pos_pred = self.score(z[:, :-t], z[:, t:])  # score for positive frame
            preds[t].append(pos_pred)

            for _ in range(self.hp.n_negatives):
                if self.training:
                    time_reorder = torch.randperm(pos_pred.shape[1])
                    batch_reorder = torch.arange(pos_pred.shape[0])
                    if self.hp.batch_shuffle:
                        batch_reorder = torch.randperm(pos_pred.shape[0])
                else:
                    time_reorder = torch.arange(pos_pred.shape[1])
                    batch_reorder = torch.arange(pos_pred.shape[0])

                neg_pred = self.score(z[:, :-t], z[batch_reorder][:, time_reorder])  # score for negative random frame
                preds[t].append(neg_pred)

        return preds

    def loss(self, preds, lengths):
        loss = 0
        for t, t_preds in preds.items():
            mask = length_to_mask(lengths - t)
            out = torch.stack(t_preds, dim=-1)
            out = F.log_softmax(out, dim=-1)
            out = out[..., 0] * mask
            loss += -out.mean()
        return loss
