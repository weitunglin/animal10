import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        # self.pool2 = nn.MaxPool2d(4, 4)
        self.conv1 = nn.Conv2d(3, 256, 5)
        self.conv2 = nn.Conv2d(256, 128, 5)
        self.conv3 = nn.Conv2d(128, 32, 5)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(128)
        # self.conv4 = nn.Conv2d(32, 32, 5)

        self.fc1 = nn.Linear(2592, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # x = self.pool2(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.batch_norm1(F.relu(self.fc1(x)))
        x = self.batch_norm2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=2, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 32, 3, padding=1),
            nn.MaxPool2d(5),
            nn.Flatten(),
            nn.Linear(32*5*5, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x
