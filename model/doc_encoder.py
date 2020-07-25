import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import AdditiveAttention
from transformers import ElectraModel


class DocEncoder(nn.Module):
    def __init__(self, hparams, weight=None) -> None:
        super(DocEncoder, self).__init__()
        self.hparams = hparams
        if weight is None:
            self.embedding = nn.Embedding(100, 300)
        else:
            # self.embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
            self.bert = ElectraModel.from_pretrained(weight)

    def forward(self, x):
        x = F.dropout(self.bert(x)[0], 0.2)[:, 0, :]
        return x
