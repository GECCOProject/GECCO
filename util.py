import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
# transforms
from torchvision import transforms as T

from torch import sigmoid
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.conv import TransformerConv
from torch.utils.data import DataLoader 
from matplotlib.image import imsave
import torch_geometric
import torchvision
from torchvision import datasets, transforms
from model import Model

def setup_model(device, model_path):
    torch.manual_seed(0)
    model = Net()
    model.load_state_dict(torch.load(model_path))
    print("Loaded the parameters for the model from %s"%model_path)
    model.to(device)
    return model

def load_checkpoint(device, checkpoint_path):
    torch.manual_seed(0)
    checkpoint = torch.load(checkpoint_path)
    model = Model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    epoch_loss = checkpoint['epoch_loss']
    return model, optimizer, epoch, epoch_loss

def load_model(device, model_path):
    torch.manual_seed(0)
    model = Model()
    model = torch.load(model_path)
    print("Loaded the parameters for the model from %s"%model_path)
    model.to(device)
    return model

def new_model(device):
    torch.manual_seed(0)
    model = Model()
    model.to(device)
    return model

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    trainset = torchvision.datasets.MNIST(root='./MNIST/', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                                shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='/data/jacob/MNIST/', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                                shuffle=False, num_workers=2)
    return trainloader, testloader
    
def load_fashion_mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    trainset = torchvision.datasets.FashionMNIST(root='./FashionMNIST/', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                                shuffle=True, num_workers=2)
    testset = torchvision.datasets.FashionMNIST(root='/data/jacob/FashionMNIST/', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                                shuffle=False, num_workers=2)
    return trainloader, testloader
