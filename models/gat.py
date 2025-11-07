import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp

class GlobalMaxPooling1D(nn.Module):
    def __init__(self):
        super(GlobalMaxPooling1D, self).__init__()

    def forward(self, x):
        return torch.max(x, dim = -1)[0]

# GAT model + transposed Conv1D input
class GATNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25, embed_dim=128, 
                 output_dim=128, dropout=0.2, conv_layers=[32, 64, 96], kernel_size=16, plm_layers = None):
        super(GATNet, self).__init__()

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)

        self.conv_layers = conv_layers       
        self.kernel_size = kernel_size
        self.stride = 1
        self.embed_dim = embed_dim

        if len(self.conv_layers) < 1:
            raise ValueError("conv_layers must have at least one element (first output size)")
        
        self.conv_xt_layers = nn.ModuleList()
        self.bn_xt_layers = nn.ModuleList()
        prev_dim = self.embed_dim
        for out_dim in self.conv_layers:
            self.conv_xt_layers.append(nn.Conv1d(in_channels=prev_dim, out_channels=out_dim, kernel_size=self.kernel_size, stride=self.stride))
            self.bn_xt_layers.append(nn.BatchNorm1d(out_dim))
            prev_dim = out_dim
        
        self.fc_xt = nn.Linear(self.conv_layers[-1], output_dim)
        self.gmp_xt = GlobalMaxPooling1D()
        self.bn_fc = nn.BatchNorm1d(output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

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
        target = data.target_encoding
        embedded_xt = self.embedding_xt(target)
        embedded_xt = torch.permute(embedded_xt, (0, 2, 1))

        xt = embedded_xt
        for conv, bn in zip(self.conv_xt_layers, self.bn_xt_layers):
            xt = conv(xt)
            xt = bn(xt)
            xt = self.relu(xt)

        xt = self.gmp_xt(xt)

        # flatten
        # xt = xt.view(-1, 96 * 811)

        # linear
        xt = self.fc_xt(xt)
        xt = self.bn_fc(xt)
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
