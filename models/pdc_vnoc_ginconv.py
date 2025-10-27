import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class GlobalMaxPooling1D(nn.Module):
    def __init(self):
        super(GlobalMaxPooling1D, self).__init__()

    def forward(self, x):
        return torch.max(x, dim = -1)[0]

# GINConv model + protein-drug-drug concatenation + inverted convolution
class PDC_Vnoc_GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, num_layers=3, kernel_size=16):

        super(PDC_Vnoc_GINConvNet, self).__init__()

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

        # protein embedding
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        
        # 1D convolution on transposed protein-drug concatenated sequence
        self.num_layers = num_layers
        if self.num_layers not in [1, 2, 3]:
            raise ValueError("num_layers must be between 1 and 3")
        
        self.kernel_size = kernel_size
        self.stride = 1

        if self.num_layers == 1:
            self.conv_xc1 = nn.Conv1d(in_channels=2*embed_dim, out_channels=n_filters, kernel_size=self.kernel_size, stride=self.stride)
            self.bn_xc1 = nn.BatchNorm1d(n_filters)

        if self.num_layers == 2:
            self.conv_xc1 = nn.Conv1d(in_channels=2*embed_dim, out_channels=n_filters, kernel_size=self.kernel_size, stride=self.stride)
            self.bn_xc1 = nn.BatchNorm1d(n_filters)
            self.conv_xc2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=self.kernel_size, stride=self.stride)
            self.bn_xc2 = nn.BatchNorm1d(n_filters)

        if self.num_layers == 3:
            self.conv_xc1 = nn.Conv1d(in_channels=2*embed_dim, out_channels=n_filters, kernel_size=self.kernel_size, stride=self.stride)
            self.bn_xc1 = nn.BatchNorm1d(n_filters)
            self.conv_xc2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=self.kernel_size, stride=self.stride)
            self.bn_xc2 = nn.BatchNorm1d(n_filters)
            self.conv_xc3 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=self.kernel_size, stride=self.stride)
            self.bn_xc3 = nn.BatchNorm1d(n_filters)

        self.gmp_xc = GlobalMaxPooling1D()
        self.fc_xc = nn.Linear(n_filters, output_dim)
        self.bn_fc = nn.BatchNorm1d(output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target_encoding

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

        # protein embedding
        embedded_xt = self.embedding_xt(target)

        # protein-drug concatenation
        x_2d = x.unsqueeze(1).expand(-1, embedded_xt.size(1), -1)
        xc = torch.cat((embedded_xt, x_2d), 2)

        # dimension permutation
        xc = torch.permute(xc, (0, 2, 1))
                                    
        # transposed convolution
        if self.num_layers == 1:
            conv_xc = self.conv_xc1(xc)
            conv_xc = self.bn_xc1(conv_xc)
            conv_xc = self.relu(conv_xc)

        elif self.num_layers == 2:
            conv_xc = self.conv_xc1(xc)
            conv_xc = self.bn_xc1(conv_xc)
            conv_xc = self.relu(conv_xc)

            conv_xc = self.conv_xc2(conv_xc)
            conv_xc = self.bn_xc2(conv_xc)
            conv_xc = self.relu(conv_xc)

        elif self.num_layers == 3:
            conv_xc = self.conv_xc1(xc)
            conv_xc = self.bn_xc1(conv_xc)
            conv_xc = self.relu(conv_xc)

            conv_xc = self.conv_xc2(conv_xc)
            conv_xc = self.bn_xc2(conv_xc)
            conv_xc = self.relu(conv_xc)
            
            conv_xc = self.conv_xc3(conv_xc)
            conv_xc = self.bn_xc3(conv_xc)
            conv_xc = self.relu(conv_xc)

        # global max pooling
        xc = self.gmp_xc(conv_xc)

        # linear
        xc = self.fc_xc(xc)
        xc = self.bn_fc(xc)
        xc = self.relu(xc)

        # protein-drug-drug concatenation
        xc = torch.cat((x, xc), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
