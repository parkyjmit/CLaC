import torch_geometric
from transformers import AutoModel


def load_graph_encoder(cfg):
    if cfg.model_name == 'gat':
        return GATEncoder(cfg)
    elif cfg.model_name == 'gcn':
        return GCNEncoder(cfg)
    elif cfg.model_name == 'graphormer':
        return GraphormerEncoder(cfg)
    else:
        raise NotImplementedError
    
def GATEncoder(cfg):
    pass

def GCNEncoder(cfg):
    pass

def GraphormerEncoder(cfg):
    model = AutoModel.from_pretrained(cfg.model_link)
    return model

