from turtle import forward
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3)
        self.pooling = nn.MaxPool2d(2)
        self.linear = nn.Linear(48, 10)
        #TODO: bn layer

    def forward(self, x):
        x = torch.relu(self.pooling(self.conv1(x))) # [14, 14, 10]
        x = torch.relu(self.pooling(self.conv2(x))) # [7, 7, 20]
        x = x.view(-1, 48) # [7*7*20]
        self.linear(x)
        return x
