from transformers import AutoModel


def load_text_encoder(cfg):
    model = AutoModel.from_pretrained(cfg.model_link)
    return model