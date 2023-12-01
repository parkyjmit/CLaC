import torch_geometric
from transformers import AutoModel
import torch
from torch import nn
from torch_geometric import nn as gnn
from torch_geometric.utils import add_self_loops
from model.utils import build_mlp, RBFExpansion

    
class GATEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = gnn.GATConv(
            in_channels=cfg.in_channels,
            out_dim=cfg.out_dim,
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

        x = self.embedding(x)  # node feature, (num_nodes, atom_fea_len)
        e = torch.sqrt(torch.sum(torch.pow(edge_attr, 2), dim=-1))
        e = self.rbf(e)  # edge feature, (num_edges, nbr_fea_len)
        for conv in self.convs:
            x = nn.functional.softplus(conv(x, edge_index, e))
        x = self.global_pool(x, batch)
        x = self.conv_to_fc(x)
        return x


class PaiNN(nn.Module):
    '''
    https://github.com/MaxH1996/PaiNN-in-PyG/blob/main/
    '''    
    def __init__(
        self,
        orig_atom_fea_len,
        num_feat,
        out_dim,
        cut_off=5.0,
        n_rbf=20,
        num_interactions=3,
    ):
        super(PaiNN, self).__init__()
        """PyG implementation of PaiNN network of Schütt et. al. Supports two arrays
           stored at the nodes of shape (num_nodes,num_feat,1) and (num_nodes, num_feat,3). For this
           representation to be compatible with PyG, the arrays are flattened and concatenated.
           Important to note is that the out_dim must match number of features"""

        self.embedding = nn.Linear(orig_atom_fea_len, num_feat)
        self.num_interactions = num_interactions
        self.cut_off = cut_off
        self.n_rbf = n_rbf
        self.num_feat = num_feat
        self.out_dim = out_dim
        self.lin1 = nn.Linear(num_feat, num_feat)
        self.lin2 = nn.Linear(num_feat, out_dim)
        self.silu = nn.functional.silu

        self.list_message = nn.ModuleList(
            [
                MessagePassPaiNN(num_feat, num_feat, cut_off, n_rbf)
                for _ in range(self.num_interactions)
            ]
        )
        self.list_update = nn.ModuleList(
            [
                UpdatePaiNN(num_feat, num_feat)
                for _ in range(self.num_interactions)
            ]
        )
        self.global_pool = gnn.global_add_pool

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        s = self.embedding(x)  # node feature, (num_nodes, atom_fea_len)
        v = torch.zeros((s.shape[0], s.shape[1], 3)).to(s.device)
        for i in range(self.num_interactions):

            s_temp, v_temp = self.list_message[i](s, v, edge_index, edge_attr)
            s, v = s_temp + s, v_temp + v
            s_temp, v_temp = self.list_update[i](s, v)
            s, v = s_temp + s, v_temp + v

        s = self.lin1(s)
        s = self.silu(s)
        s = self.lin2(s)
        s = self.global_pool(s, batch)

        return s


class MessagePassPaiNN(gnn.MessagePassing):
    def __init__(self, num_feat, out_dim, cut_off=5.0, n_rbf=20):
        super(MessagePassPaiNN, self).__init__(aggr="add")

        self.lin1 = nn.Linear(num_feat, out_dim)
        self.lin2 = nn.Linear(out_dim, 3 * out_dim)
        self.lin_rbf = nn.Linear(n_rbf, 3 * out_dim)
        self.silu = nn.functional.silu

        self.RBF = BesselBasis(cut_off, n_rbf)
        self.f_cut = CosineCutoff(cut_off)
        self.num_feat = num_feat

    def forward(self, s, v, edge_index, edge_attr):

        s = s.flatten(-1)
        v = v.flatten(-2)

        flat_shape_v = v.shape[-1]
        flat_shape_s = s.shape[-1]

        x = torch.cat([s, v], dim=-1)

        x = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            flat_shape_s=flat_shape_s,
            flat_shape_v=flat_shape_v,
        )

        return x

    def message(self, x_j, edge_attr, flat_shape_s, flat_shape_v):

        # Split Input into s_j and v_j
        s_j, v_j = torch.split(x_j, [flat_shape_s, flat_shape_v], dim=-1)

        # r_ij channel
        rbf = self.RBF(edge_attr)
        ch1 = self.lin_rbf(rbf)
        cut = self.f_cut(edge_attr.norm(dim=-1))
        W = torch.einsum("ij,i->ij", ch1, cut)  # ch1 * f_cut

        # s_j channel
        phi = self.lin1(s_j)
        phi = self.silu(phi)
        phi = self.lin2(phi)

        # Split

        left, dsm, right = torch.split(phi * W, self.num_feat, dim=-1)

        # v_j channel
        normalized = nn.functional.normalize(edge_attr, p=2, dim=1)
        v_j = v_j.reshape(-1, int(flat_shape_v / 3), 3)
        hadamard_right = torch.einsum("ij,ik->ijk", right, normalized)
        hadamard_left = torch.einsum("ijk,ij->ijk", v_j, left)
        dvm = hadamard_left + hadamard_right

        # Prepare vector for update
        x_j = torch.cat((dsm, dvm.flatten(-2)), dim=-1)

        return x_j

    def update(self, out_aggr, flat_shape_s, flat_shape_v):

        s_j, v_j = torch.split(out_aggr, [flat_shape_s, flat_shape_v], dim=-1)

        return s_j, v_j.reshape(-1, int(flat_shape_v / 3), 3)


class UpdatePaiNN(torch.nn.Module):
    def __init__(self, num_feat, out_dim):
        super(UpdatePaiNN, self).__init__()

        self.lin_up = nn.Linear(2 * num_feat, out_dim)
        self.denseU = nn.Linear(num_feat, out_dim, bias=False)
        self.denseV = nn.Linear(num_feat, out_dim, bias=False)
        self.lin2 = nn.Linear(out_dim, 3 * out_dim)
        self.silu = nn.functional.silu
        self.num_feat = num_feat

    def forward(self, s, v):

        s = s.flatten(-1)
        v = v.flatten(-2)

        flat_shape_v = v.shape[-1]
        flat_shape_s = s.shape[-1]

        v_u = v.reshape(-1, int(flat_shape_v / 3), 3)
        v_ut = torch.transpose(
            v_u, 1, 2
        )  # need transpose to get lin.comb a long feature dimension
        U = torch.transpose(self.denseU(v_ut), 1, 2)
        V = torch.transpose(self.denseV(v_ut), 1, 2)

        # form the dot product
        UV = torch.einsum("ijk,ijk->ij", U, V)

        # s_j channel
        nV = torch.norm(V, dim=-1)

        s_u = torch.cat([s, nV], dim=-1)
        s_u = self.lin_up(s_u)
        s_u = nn.functional.silu(s_u)
        s_u = self.lin2(s_u)
        # s_u = Func.silu(s_u)

        # final split
        top, middle, bottom = torch.split(s_u, self.num_feat, dim=-1)

        # outputs
        dvu = torch.einsum("ijk,ij->ijk", v_u, top)
        dsu = middle * UV + bottom

        return dsu, dvu.reshape(-1, int(flat_shape_v / 3), 3)



class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        # self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
        self.cutoff = cutoff

    def forward(self, distances):
        """Compute cutoff.

        Args:
            distances (torch.Tensor): values of interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(distances * torch.pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs


class BesselBasis(torch.nn.Module):
    """
    Sine for radial basis expansion with coulomb decay. (0th order Bessel from DimeNet)
    """

    def __init__(self, cutoff=5.0, n_rbf=None):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super(BesselBasis, self).__init__()
        # compute offset and width of Gaussian functions
        freqs = torch.arange(1, n_rbf + 1) * torch.pi / cutoff
        self.register_buffer("freqs", freqs)

    def forward(self, inputs):
        inputs = torch.norm(inputs, p=2, dim=1)
        a = self.freqs
        ax = torch.outer(inputs, a)
        sinax = torch.sin(ax)

        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm[:, None]

        return y
    

class MolCLR(nn.Module):
    def __init__(self, out_dim) -> None:
        super().__init__()
        num_layer = 5                          # number of graph conv layers
        emb_dim = 300                          # embedding dimension in graph conv layers
        feat_dim = 512                         # output feature dimention
        drop_ratio = 0                         # dropout ratio
        pool = 'mean'                            # readout pooling (i.e., mean/max/add)
        self.model = GINet(num_layer, emb_dim, feat_dim, drop_ratio, pool)
        self.model.load_state_dict(torch.load('/home/yj/PycharmProjects/MIT/CLaMP/model/molclr-gin.pth'))

    def forward(self, data):
        return self.model(data)[0]

num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 

class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool='mean'):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        
        if pool == 'mean':
            self.pool = gnn.global_mean_pool
        elif pool == 'max':
            self.pool = gnn.global_max_pool
        elif pool == 'add':
            self.pool = gnn.global_add_pool
        
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim//2)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = nn.functional.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = nn.functional.dropout(nn.functional.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        out = self.out_lin(h)
        
        return h, out


class GINEConv(gnn.MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)