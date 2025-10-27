import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, AttentionalAggregation, GlobalAttention
from torch_geometric.nn import global_max_pool as gmp
from torch_scatter import scatter_add


class DGL_WeightAndSum(nn.Module):
    def __init__(self, in_feats):
        super(DGL_WeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1), 
            nn.Sigmoid()
        )

    def forward(self, x, batch):
        w = self.atom_weighting(x)
        xw = x * w
        x_sum = scatter_add(xw, batch, dim=0)
        return x_sum

class DGLLife_WeightedSumAndMax(nn.Module):
    def __init__(self, in_feats):
        super(DGLLife_WeightedSumAndMax, self).__init__()
        self.weight_and_sum = DGL_WeightAndSum(in_feats)

    def forward(self, x, batch):
        h_g_sum = self.weight_and_sum(x, batch)
        h_g_max = gmp(x, batch)
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        return h_g

# Re-implementation of PGraphDTA GAT model with ESM protein representation
class ESM_GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=25,
                     n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, num_layers=None, kernel_size=None):
        super(ESM_GATNet, self).__init__()

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, heads=1, dropout=dropout)
        
        # self.attention = GlobalAttention(gate_nn=nn.Linear(output_dim, 1))
        # self.attention = AttentionalAggregation(gate_nn=nn.Linear(output_dim, 1))

        self.readout = DGLLife_WeightedSumAndMax(output_dim)

        # ESM protein embedding linear layer
        self.fc_xt = nn.Linear(1280, 256)

        # combined layers
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target_embedding

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)

        x = self.readout(x, batch)          # weighted sum & max pooling

        # x_sum = self.attention(x, batch)
        # x_max = gmp(x, batch)
        # x = torch.cat([x_sum, x_max], dim=1)

        # ESM Protein embedding linear layer learning
        xt = self.fc_xt(target)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
