import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GINConv model + protein language model representation
class PLM_GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25, embed_dim=320,
                output_dim=128, dropout=0.2, conv_layers=None, kernel_size=None, plm_layers = [128]):

        super(PLM_GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

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
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

        # print(self)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target_embedding

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

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
