import torch
import torch.nn as nn
import torch.nn.functional as F
from model.doc_encoder import DocEncoder
from model.attention import  AdditiveAttention


class NRMS(nn.Module):
    def __init__(self, hparams, weight=None):
        super(NRMS, self).__init__()
        self.hparams = hparams
        self.doc_encoder = DocEncoder(hparams, weight=weight)
        self.mha = nn.MultiheadAttention(hparams['embed_size'], hparams['nhead'], dropout=0.1)
        self.proj = nn.Linear(hparams['embed_size'], hparams['embed_size'])
        self.additive_attn = AdditiveAttention(hparams['embed_size'], hparams['v_size'])
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, clicks, cands, labels=None):
        """forward

        Args:
            clicks (tensor): [num_user, num_click_docs, seq_len]
            cands (tensor): [num_user, num_candidate_docs, seq_len]
        """
        num_click_docs = clicks.shape[1]
        num_cand_docs = cands.shape[1]
        num_user = clicks.shape[0]
        seq_len = clicks.shape[2]
        clicks = clicks.reshape(-1, seq_len)
        cands = cands.reshape(-1, seq_len)
        click_embed = self.doc_encoder(clicks)
        cand_embed = self.doc_encoder(cands)
        click_embed = click_embed.reshape(num_user, num_click_docs, -1)
        cand_embed = cand_embed.reshape(num_user, num_cand_docs, -1)
        click_embed = click_embed.permute(1, 0, 2)
        click_output, _ = self.mha(click_embed, click_embed, click_embed)
        click_output = F.dropout(click_output.permute(1, 0, 2), 0.2)

        click_repr = self.proj(click_output)
        click_repr, _ = self.additive_attn(click_output)
        logits = torch.bmm(click_repr.unsqueeze(1), cand_embed.permute(0, 2, 1)).squeeze(1) # [B, 1, hid], [B, 10, hid]
        if labels is not None:
            loss = self.criterion(logits, labels)
            return loss, logits
        return torch.sigmoid(logits)
        # return torch.softmax(logits, -1)
