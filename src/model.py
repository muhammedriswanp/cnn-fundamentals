import torch.nn as nn

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3
        )
        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(
            stride=2,
            kernel_size=2
        )

        self.flatten = nn.Flatten()


        self.fc1 = nn.Linear(5408, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x= self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x