hparams = {
    'batch_size': 32,
    'lr': 2e-5,
    'name': 'bert',
    'version': 'v1',
    'description': 'NRMS BERT doc_encoder',
    'pretrained_model': 'hfl/chinese-roberta-wwm-ext',
    'model': {
        'dct_size': 21128,
        'nhead': 16,
        'embed_size': 768,
        'encoder_size': 768,
        'v_size': 300
    },
    'data': {
        'pos_k': 50,
        'neg_k': 4,
        'maxlen': 15
    }
}
