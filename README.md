# NRMS

* Pytorch 1.5
* [Neural News Recommendation with Multi-Head Self-Attention](https://github.com/wuch15/IJCAI2019-NAML)
* [Offical Implementation](https://github.com/wuch15/IJCAI2019-NAML) (keras)

## Change

* Use Range instead of Adam Optimizer
* title encoder: branch for RoBERTa and ELETRA-Small 
* pytorch-lightning
    * tensorboard support
    * early stop


## Benchmark

* Use Taiwan PTT forum as data (Tranditional Chinese)
* regard a comment as the user intereseted in the post
* train on one Titan RTX
* train until early stop

### Model

* original: Use Word2Vec pretrained on Wiki-zh
* Roberta: roberta-base in [this](https://github.com/ymcui/Chinese-BERT-wwm)
* ELETRA: electra-smiall in [this](https://github.com/ymcui/Chinese-ELECTRA)

### training time

* original(Adam): 1 hr
* original(Ranger): 1 hr 4 min
* Roberta(Ranger): 19 hr 46 min
* ELETRA-Small(Ranger): 2hr 19min

### Score on ValidationSet


#### AUROC

* original(Adam): 0.86
* original(Ranger): 0.89
* Roberta(Ranger): 0.94
* ELETRA-small(Ranger): 0.91

#### ndcg@5

* original(Adam): 0.73
* original(Ranger): 0.79
* Roberta(Ranger): 0.88
* ELETRA-small(Ranger): 0.81

#### ndcg@10

* original(Adam): 0.67
* original(Ranger): 0.72
* Roberta(Ranger): 0.81
* ELETRA-small(Ranger): 0.74