import torch_geometric
from transformers import AutoModel
from torch import nn
from torch_geometric import nn as gnn


def load_graph_encoder(cfg):
    if cfg.model_name == 'gat':
        return GATEncoder(cfg)
    elif cfg.model_name == 'gcn':
        return GCNEncoder(cfg)
    elif cfg.model_name == 'graphormer':
        return AutoModel.from_pretrained(cfg.model_link)
    else:
        raise NotImplementedError
    
class GATEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = gnn.GATConv(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            heads=cfg.heads,
            dropout=cfg.dropout
        )

class GCNEncoder(nn.Module):
    pass
