import random
from typing import List

import orjson as json
import torch
from gensim.models import Word2Vec
from torch.utils import data
from tqdm import tqdm


class Dataset(data.Dataset):
    def __init__(self, article_file: str, user_file: str, w2v, maxlen: int = 15, pos_num: int = 50, neg_k: int = 4):
        self.articles = self.load_json(article_file)
        self.users = self.load_json(user_file)
        self.maxlen = maxlen
        self.neg_k = neg_k
        self.pos_num = pos_num

        self.w2id = {w: w2v.wv.vocab[w].index for w in w2v.wv.vocab}

    def load_json(self, file: str):
        with open(file, 'r') as f:
            return json.loads(f.read())

    def sent2idx(self, tokens: List[str]):
        # tokens = tokens[3:]
        if ']' in tokens:
            tokens = tokens[tokens.index(']'):]
        tokens = [self.w2id[token.strip()]
                  for token in tokens if token.strip() in self.w2id.keys()]
        tokens += [0] * (self.maxlen - len(tokens))
        tokens = tokens[:self.maxlen]
        return tokens

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx: int):
        """getitem

        Args:
            idx (int): 
        Data:
            return (
                user_id (int): 1
                click (tensor): [batch, num_click_docs, seq_len]
                cand (tensor): [batch, num_candidate_docs, seq_len]
                label: candidate docs label (0 or 1)
            )
        """
        push = self.users[idx]['push']
        random.shuffle(push)
        push = push[:self.pos_num]
        uid = self.users[idx]['user_id']
        click_doc = [self.sent2idx(self.articles[p]['title']) for p in push]
        cand_doc = []
        cand_doc_label = []
        # neg
        for i in range(self.neg_k):
            neg_id = -1
            while neg_id == -1 or neg_id in push:
                neg_id = random.randint(0, len(self.articles) - 1)
            cand_doc.append(self.sent2idx(self.articles[neg_id]['title']))
            cand_doc_label.append(0)
        # pos
        try:
            cand_doc.append(self.sent2idx(
                self.articles[push[random.randint(50, len(self.push) - 1)]['title']]))
            cand_doc_label.append(1)
        except Exception:
            try:
                cand_doc.append(self.sent2idx(self.articles[push[0]]['title']))
            except:
                print(push[0])
                print(self.articles[push[0]])
            cand_doc_label.append(1)

        tmp = list(zip(cand_doc, cand_doc_label))
        random.shuffle(tmp)
        cand_doc, cand_doc_label = zip(*tmp)
        return torch.tensor(click_doc), torch.tensor(cand_doc), torch.tensor(cand_doc_label, dtype=torch.float).argmax(0)

class ValDataset(Dataset):
    def __init__(self, num=50, *args, **kwargs) -> None:
        super(ValDataset, self).__init__(*args, **kwargs)
        self.num = num
    
    def __getitem__(self, idx: int):
        push = self.users[idx]['push']
        random.shuffle(push)
        uid = self.users[idx]['user_id']
        click_doc = [self.sent2idx(self.articles[p]['title']) for p in push[:self.pos_num]]
        
        true_num = 10
        # true_num = random.randint(1, min(self.num, len(push)) )
        f_num = self.num - true_num
        cand_doc = random.sample(push, true_num) # true
        cand_doc_label = [1] * true_num
        cand_doc.extend(random.sample(range(0, len(self.articles)), f_num)) # false
        cand_doc_label.extend([0] * f_num)
        tmp = list(zip(cand_doc, cand_doc_label))
        random.shuffle(tmp)
        cand_doc, cand_doc_label = zip(*tmp)
        cand_doc = [self.sent2idx(self.articles[cand]['title']) for cand in cand_doc]
        return torch.LongTensor(click_doc), torch.LongTensor(cand_doc), torch.LongTensor(cand_doc_label)


if __name__ == '__main__':
    w2v = Word2Vec.load('./word2vec/wiki_300d_5ws.model')
    ds = ValDataset(50, './data/articles.json', './data/users_list.json',
                 w2v, maxlen=30, pos_num=50, neg_k=4)
    print(ds[10])
    for i in tqdm(ds):
        pass
