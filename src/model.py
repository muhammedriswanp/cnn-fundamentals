import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, kernel_size=3, stride=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()

        # dynamically calculate fc1 input size
        dummy = torch.zeros(1, 1, 28, 28)
        fc1_input = self.flatten(self.features(dummy)).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(fc1_input, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x