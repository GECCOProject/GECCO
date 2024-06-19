import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import sigmoid
from torch_geometric.nn import ResGatedGraphConv
from torch_geometric.data import Data, Batch
from mygat import Att

IMG_SIZE = 28
OUT = 10
featurelength = 64
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")     

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Att(IMG_SIZE*IMG_SIZE, featurelength)
        self.conv2 = ResGatedGraphConv(featurelength, featurelength)

        self.fc1 = nn.Linear(featurelength // 2, OUT)
        self.bn = nn.BatchNorm1d(featurelength)

        self.dropout = nn.Dropout(0.15)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x, train):
        batch_size = x.shape[0]

        x_1 = torch.reshape(x, [batch_size, IMG_SIZE*IMG_SIZE])

        edge_index = torch.tensor([[i for i in range(batch_size)] for j in range(batch_size)]).to(device)
        edge_index = edge_index.reshape(1, -1)
        edge_index = torch.cat((edge_index, torch.flip(edge_index, [0])), dim=0)

        x_2 = self.conv1(x_1)
        
        if train:
            x_2 = self.dropout(x_2) 
            
        x_3 = F.relu(x_2)

        x_4 = self.conv2(x_2, edge_index)
    
        x_4 = F.relu(x_4)
        x_4 = self.pool(self.bn(x_4))

        attention = torch.matmul(x_4, torch.transpose(x_4, 0, 1))
        attention = sigmoid(attention)
        attention = attention / torch.sum(attention, dim=1).unsqueeze(1)

        x_5 = torch.matmul(attention, x_4)


        x_6 = x_5 + x_4
        output = self.fc1(x_6)
        return output
