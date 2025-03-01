import torch
from torch import nn
from torch_geometric.data import Data


class GraphPerturbationAugmentation(nn.Module):
    def __init__(self, perturbation):
        super().__init__()
        self.perturbation = perturbation

    @torch.no_grad()
    def forward(self, graph: Data) -> Data:
        graph = graph.clone()
        graph.edge_attr = graph.edge_attr + torch.randn_like(graph.edge_attr) * self.perturbation  # mean 0, std sfg.perturbation
        return graph
    

class GraphAttrMaskingAugmentation(nn.Module):
    def __init__(self, mask_prob):
        super().__init__()
        self.mask_prob = mask_prob

    @torch.no_grad()
    def forward(self, graph: Data) -> Data:
        graph = graph.clone()
        # mask node and edge as 0 with probability cfg.mask_prob
        node_mask = torch.rand(graph.num_nodes) < self.mask_prob
        graph.x[node_mask] = 0.0
        edge_mask = torch.rand(graph.num_edges) < self.mask_prob
        graph.edge_attr[edge_mask] = 0.0
        return graph


class TokenRandomMaskingAugmentation(nn.Module):
    def __init__(self, mask_prob, mask_token):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_token = mask_token

    @torch.no_grad()
    def forward(self, texts: dict) -> torch.Tensor:
        input_ids = texts['input_ids'].clone()
        # mask token as <masked> token with probability cfg.mask_prob
        mask = torch.rand(input_ids.shape) < self.mask_prob
        input_ids[mask] = self.mask_token
        # set labels as original tokens
        labels = texts["input_ids"].clone()
        labels[input_ids != self.mask_token] = -100
        # update texts
        texts['labels'] = labels
        texts['input_ids'] = input_ids
        return texts
    

