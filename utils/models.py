import torch
import torch.nn as nn
import torch.nn.functional as F


# Network architecture
class LinearNet(nn.Module):

    # Constructor
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    # Forward pass
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        output = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output