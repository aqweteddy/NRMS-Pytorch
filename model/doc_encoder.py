import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import  AdditiveAttention


class DocEncoder(nn.Module):
    def __init__(self, hparams, weight=None) -> None:
        super(DocEncoder, self).__init__()
        self.hparams = hparams
        if weight is None:
            self.embedding = nn.Embedding(100, 300)
        else:
            self.embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
        self.mha = nn.MultiheadAttention(hparams['embed_size'], num_heads=hparams['nhead'], dropout=0.1)
        self.proj = nn.Linear(hparams['embed_size'], hparams['encoder_size'])
        self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])
    
    def forward(self, x):
        x = F.dropout(self.embedding(x), 0.2)
        x = x.permute(1, 0, 2)
        output, _ = self.mha(x, x, x)
        output = F.dropout(output.permute(1, 0, 2))
        output = self.proj(output)
        output, _ = self.additive_attn(output)
        return output
