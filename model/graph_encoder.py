import torch_geometric
from transformers import AutoModel
import torch
from torch import nn
from torch_geometric import nn as gnn
from model.utils import build_mlp, RBFExpansion

    
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

class GraphEncoder(nn.Module):
    pass

class CGCNN(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1, out_dim=1) -> None:
        super().__init__()
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.rbf = RBFExpansion(bins=nbr_fea_len)
        self.convs = nn.ModuleList([gnn.CGConv(channels=atom_fea_len,
                                    dim=nbr_fea_len,batch_norm=True)
                                    for _ in range(n_conv)])
        self.global_pool = gnn.global_add_pool
        self.conv_to_fc = build_mlp(atom_fea_len, h_fea_len, n_h, out_dim, nn.Softplus())
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.embedding(x)
        e = torch.sqrt(torch.sum(torch.pow(edge_attr, 2), dim=-1))
        e = self.rbf(e)
        for conv in self.convs:
            x = nn.functional.softplus(conv(x, edge_index, e))
        x = self.global_pool(x, batch)
        x = self.conv_to_fc(x)
        return x
