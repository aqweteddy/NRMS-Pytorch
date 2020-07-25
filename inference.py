import pytorch_lightning as pl
from gensim.models import Word2Vec
import torch
from model import NRMS
from transformers import BertTokenizerFast


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(hparams['pretrained_model'])
        if hparams['model']['dct_size'] == 'auto':
            hparams['model']['dct_size'] = len(self.w2v.wv.vocab)
        self.model = NRMS(hparams['model'], hparams['pretrained_model'])
        self.model.eval()
        self.hparams = hparams
        self.maxlen = hparams['data']['maxlen']

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
        with torch.no_grad():
            idx, val = self(viewed_token, cands_token, topk)
        val = val.squeeze().detach().cpu().tolist()

        result = [cands[i] for i in idx.squeeze()]
        return result, val
    
    def sent2idx(self, tokens: str):
        tokens = self.tokenizer.encode(tokens)
        tokens += [0] * (self.maxlen - len(tokens))
        tokens = tokens[:self.maxlen]
        return tokens
    
    def tokenize(self, sents: str):
        return self.tokenizer.tokenize(sents)


def print_func(r):
    for t in r:
        print(''.join(t))
    
if __name__ == '__main__':
    import json, random, os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    with open('./data/articles.json', 'r') as f:
        articles = json.loads(f.read())
    with open('./data/users_list.json', 'r') as f:
        users = json.loads(f.read())
    nrms = Model.load_from_checkpoint('lightning_logs/bert/v1/epoch=10-auroc=0.92.ckpt')
    viewed = users[1001]['push'][:50]
    viewed = [''.join(articles[v]['title']) for v in viewed]
    cands = [''.join(a['title']) for a in random.sample(articles, 20)] + viewed[40:]
    print('read')
    print_func(viewed[:40])
    print('truth')
    print_func(viewed[40:])
    viewed = viewed[:40]

    result, val = nrms.predict_one(viewed, cands, 20)
    print('result')
    print_func(result)
    print(val)
