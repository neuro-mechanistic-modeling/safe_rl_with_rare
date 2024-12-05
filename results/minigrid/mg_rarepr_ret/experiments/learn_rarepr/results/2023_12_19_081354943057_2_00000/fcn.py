'''
Defines the neural network architecture
'''
import torch
from torch import nn
import torch.nn.functional as F


cuda = torch.device("cuda")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Network(nn.Module):
    def __init__(self, device, weights=[15, 64, 64, 9]):
        super().__init__()
        self.device = device
        # self.weights = weights
        self.layers = nn.ModuleList(
            [nn.Linear(weights[i], weights[i + 1]) for i in range(len(weights) - 1)]
        ).to(device)

    def forward(self, x):
        p = []
        for i in range(len(self.layers) - 1):
            x = self.layers[i].forward(x)
            x = F.relu(x)
        x = self.layers[-1].forward(x)
        return x
