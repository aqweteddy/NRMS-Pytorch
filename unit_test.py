import torch.nn as nn
import torch
from config import hparams

def attention():
    from model.attention import MultiheadAttentionContainer, InProjContainer, ScaledDotProduct, AdditiveAttention

    q = torch.rand(10, 16, 200)
    k = v = torch.rand(10, 16, 200)

    print('MultiHead: ')
    proj = InProjContainer(nn.Linear(200, 200), nn.Linear(
        200, 200), nn.Linear(200, 200))
    mha = MultiheadAttentionContainer(nhead=8,
                                      in_proj_container=proj,
                                      attention_layer=ScaledDotProduct(),
                                      out_proj=nn.Linear(200, 200))
    output, score = mha(q, k, v)
    print(output.shape)
    print('Additive: ')
    attn = AdditiveAttention(200, 300)
    output = output.permute(1, 0, 2)
    output, score = attn(output)
    print(output.size(), score.size())

def doc_encoder():
    from model.doc_encoder import DocEncoder

    encoder = DocEncoder(hparams['model'])
    x = torch.randint(0, 100, (16, 100))
    output = encoder(x)
    print(output.shape)

def NRMS():
    from model.net import NRMS

    nrms = NRMS(hparams['model'])
    clicks = torch.randint(0, 100, (8, 50, 100))
    cands = torch.randint(0, 100, (8, 10, 100))
    logits = nrms(clicks, cands)
    # print(logits.shape)

# doc_encoder()
attention()
# NRMS()