import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class GlobalMaxPooling1D(nn.Module):
    def __init__(self):
        super(GlobalMaxPooling1D, self).__init__()

    def forward(self, x):
        return torch.max(x, dim = -1)[0]

# GCN-CNN based model + transposed Conv1D input
class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25, embed_dim=128, 
                 output_dim=128, dropout=0.2, conv_layers=[32, 64, 96], kernel_size=16, plm_layers = None):

        super(GAT_GCN, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

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
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target_encoding
        # print('x shape = ', x.shape)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

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
