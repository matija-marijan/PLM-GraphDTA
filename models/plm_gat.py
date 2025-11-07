import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp


# GAT model + protein language model representation
class PLM_GATNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25, embed_dim=320,
                output_dim=128, dropout=0.2, conv_layers=None, kernel_size=None, plm_layers = [128]):
        super(PLM_GATNet, self).__init__()

        self.n_output = n_output
        
        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # PLM embedding linear layers built dynamically
        # embed_dim is the input size (target_embedding dim)
        # plm_layers defines successive output sizes; plm_layers[-1] is final projection size
        self.plm_layers = plm_layers
        self.embed_dim = embed_dim
        
        if len(self.plm_layers) < 1:
            raise ValueError("plm_layers must have at least one element (first output size)")

        self.fc_xt_layers = nn.ModuleList()
        self.bn_xt_layers = nn.ModuleList()
        prev_dim = self.embed_dim
        for out_dim in self.plm_layers:
            self.fc_xt_layers.append(nn.Linear(prev_dim, out_dim))
            self.bn_xt_layers.append(nn.BatchNorm1d(out_dim))
            prev_dim = out_dim

        # combined layers
        # Input to fc1 is concatenation of graph (output_dim) and PLM branch (self.plm_layers[-1])
        self.fc1 = nn.Linear(output_dim + self.plm_layers[-1], 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        # protein input feed-forward:
        target = data.target_embedding
        # PLM embedding linear layers
        xt = target
        for fc, bn in zip(self.fc_xt_layers, self.bn_xt_layers):
            xt = fc(xt)
            xt = bn(xt)
            xt = self.relu(xt)

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
