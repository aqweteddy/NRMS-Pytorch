import pytorch_lightning as pl
from gensim.models import Word2Vec
import torch
from model import NRMS
from typing import List
from gaisTokenizer import Tokenizer


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.w2v = Word2Vec.load(hparams['pretrained_model'])
        self.w2id = {w: self.w2v.wv.vocab[w].index for w in self.w2v.wv.vocab}

        if hparams['model']['dct_size'] == 'auto':
            hparams['model']['dct_size'] = len(self.w2v.wv.vocab)
        self.model = NRMS(hparams['model'], torch.tensor(self.w2v.wv.vectors))
        self.hparams = hparams
        self.maxlen = hparams['data']['maxlen']
        self.tokenizer = Tokenizer('k95763565C5F785B50546754545D77505F0325160B58173C17291B3D5E2500135001671C06272B3B06281E1E5E55A9F7EB80C0E58AD1EB50AC')

    def forward(self, viewed, cands, topk):
        """forward

        Args:
            viewed (tensor): [B, viewed_num, maxlen]
            cands (tesnor): [B, cand_num, maxlen]
        Returns:
            val: [B] 0 ~ 1
            idx: [B] 
        """
        logits = self.model(viewed, cands)
        val, idx = logits.topk(topk)
        return idx, val
    
    def predict_one(self, viewed, cands, topk):
        """predict one user

        Args:
            viewed (List[List[str]]): 
            cands (List[List[str]]): 
        Returns:
            topk of cands
        """
        viewed_token = torch.tensor([self.sent2idx(v) for v in viewed]).unsqueeze(0)
        cands_token = torch.tensor([self.sent2idx(c) for c in cands]).unsqueeze(0)
        idx, val = self(viewed_token, cands_token, topk)
        val = val.squeeze().detach().cpu().tolist()

        result = [cands[i] for i in idx.squeeze()]
        return result, val
    
    def sent2idx(self, tokens: List[str]):
        if ']' in tokens:
            tokens = tokens[tokens.index(']'):]
        tokens = [self.w2id[token.strip()]
                  for token in tokens if token.strip() in self.w2id.keys()]
        tokens += [0] * (self.maxlen - len(tokens))
        tokens = tokens[:self.maxlen]
        return tokens
    
    def tokenize(self, sents: str):
        return self.tokenizer.tokenize(sents)


def print_func(r):
    for t in r:
        print(''.join(t))
    
if __name__ == '__main__':
    import json, random
    with open('./data/articles.json', 'r') as f:
        articles = json.loads(f.read())
    with open('./data/users_list.json', 'r') as f:
        users = json.loads(f.read())
    nrms = Model.load_from_checkpoint('lightning_logs/ranger/v3/epoch=30-auroc=0.89.ckpt')
    viewed = users[1001]['push'][:50]
    viewed = [articles[v]['title'] for v in viewed]
    print_func(viewed)
    cands = [a['title'] for a in random.sample(articles, 20)] + viewed[:10]
    result, val = nrms.predict_one(viewed, cands, 20)
    print('result')
    print_func(result)
    print(val)
