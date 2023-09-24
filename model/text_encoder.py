from transformers import AutoModel


def load_text_encoder(cfg):
    if cfg.model_name == 'distilbert':
        return BERT(cfg)
    else:
        raise NotImplementedError
    
def BERT(cfg):
    model = AutoModel.from_pretrained(cfg.model_link)
    return model
