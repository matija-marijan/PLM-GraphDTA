import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GINConv model + DeepFRI protein representation
class FRI_GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, num_layers=3, kernel_size=None):

        super(FRI_GINConvNet, self).__init__()

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

        # DeepFRI protein embedding linear layer
        self.num_layers = num_layers
        if self.num_layers not in [1, 2, 3]:
            raise ValueError("num_layers must be between 1 and 3")
        if self.num_layers == 1:
            self.fc_xt = nn.Linear(4096, 128)
            self.bn_xt = nn.BatchNorm1d(128)

        elif self.num_layers == 2:
            self.fc_xt = nn.Linear(4096, 512)
            self.fc_xt2 = nn.Linear(512, 128)
            self.bn_xt = nn.BatchNorm1d(512)
            self.bn_xt2 = nn.BatchNorm1d(128)

        elif self.num_layers == 3:
            self.fc_xt = nn.Linear(4096, 512)
            self.fc_xt2 = nn.Linear(512, 256)
            self.fc_xt3 = nn.Linear(256, 128)
            self.bn_xt = nn.BatchNorm1d(512)
            self.bn_xt2 = nn.BatchNorm1d(256)
            self.bn_xt3 = nn.BatchNorm1d(128)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

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

        # DeepFRI Protein embedding linear layer learning
        if self.num_layers == 1:
            xt = self.fc_xt(target)
            xt = self.bn_xt(xt)
            xt = self.relu(xt)
        
        elif self.num_layers == 2:
            xt = self.fc_xt(target)
            xt = self.bn_xt(xt)
            xt = self.relu(xt)

            xt = self.fc_xt2(xt)
            xt = self.bn_xt2(xt)
            xt = self.relu(xt)

        elif self.num_layers == 3:
            xt = self.fc_xt(target)
            xt = self.bn_xt(xt)
            xt = self.relu(xt)

            xt = self.fc_xt2(xt)
            xt = self.bn_xt2(xt)
            xt = self.relu(xt)

            xt = self.fc_xt3(xt)
            xt = self.bn_xt3(xt)
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
