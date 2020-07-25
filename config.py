hparams = {
    'batch_size': 32,
    'lr': 3e-4,
    'name': 'electra_small',
    'version': 'v1',
    'description': 'NRMS ELECTRA small doc_encoder',
    'pretrained_model': 'hfl/chinese-electra-small-discriminator',
    'model': {
        'dct_size': 21128,
        'nhead': 16,
        'embed_size': 256,
        'encoder_size': 256,
        'v_size': 200
    },
    'data': {
        'pos_k': 50,
        'neg_k': 4,
        'maxlen': 15
    }
}
