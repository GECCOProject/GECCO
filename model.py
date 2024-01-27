import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import sigmoid
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, Batch
from mygat import Att

IMG_SIZE = 28
OUT = 10
featurelength = 32
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")     

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Att(IMG_SIZE*IMG_SIZE, featurelength * 2, heads=4)
        self.conv2 = SAGEConv(featurelength, featurelength)

        self.fc1 = nn.Linear(featurelength // 2, OUT)

        self.dropout = nn.Dropout(0.15)
        self.pool = nn.MaxPool1d(2)

        self.bn1 = nn.BatchNorm1d(featurelength)

    def forward(self, x, train):
        batch_size = x.shape[0]

        x_1 = torch.reshape(x, [batch_size, IMG_SIZE*IMG_SIZE])

        edge_index = torch.tensor([[i for i in range(batch_size)] for j in range(batch_size)]).to(device)
        edge_index = edge_index.reshape(1, -1)
        edge_index = torch.cat((edge_index, torch.flip(edge_index, [0])), dim=0)


        x_1 = self.conv1(x_1)
        x_1 = F.relu(x_1)

        if train:
            x_1 = self.dropout(x_1)
        x_2 = self.pool(x_1.unsqueeze(0)).squeeze(0)


        x_2 = self.conv2(x_2, edge_index)
        x_2 = self.bn1(x_2)
        x_2 = F.relu(x_2)

        if train:
            x_2 = self.dropout(x_2)

        x_3 = self.pool(x_2.unsqueeze(0)).squeeze(0)

        attention = torch.matmul(x_3, torch.transpose(x_3, 0, 1))
        attention = sigmoid(attention)
        attention = attention / torch.sum(attention, dim=1).unsqueeze(1)

        x_4 = torch.matmul(attention, x_3)

        residual = x_4 + x_3

        x_5 = self.fc1(residual)
        return x_5