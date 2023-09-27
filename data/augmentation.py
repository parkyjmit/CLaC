import torch
from torch import nn
from torch_geometric.data import Data


class GraphPerturbationAugmentation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def forward(self, graph: Data) -> Data:
        graph = graph.clone()
        graph.edge_attr = graph.edge_attr + torch.randn_like(graph.edge_attr) * self.cfg.perturbation  # mean 0, std sfg.perturbation
        return graph
    

class GraphAttrMaskingAugmentation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def forward(self, graph: Data) -> Data:
        graph = graph.clone()
        # mask node and edge as 0 with probability cfg.mask_prob
        node_mask = torch.rand(graph.num_nodes) < self.cfg.mask_prob
        graph.x[node_mask] = 0.0
        edge_mask = torch.rand(graph.num_edges) < self.cfg.mask_prob
        graph.edge_attr[edge_mask] = 0.0
        return graph


class TokenRandomMaskingAugmentation(nn.Module):
    def __init__(self, cfg, masked_token):
        super().__init__()
        self.cfg = cfg
        self.masked_token = masked_token

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.clone()
        # mask token as <masked> token with probability cfg.mask_prob
        mask = torch.rand(tokens.shape) < self.cfg.mask_prob
        tokens[mask] = self.masked_token
        return tokens
    

        