import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3Layer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        # self.pool2 = nn.MaxPool2d(4, 4)
        self.conv1 = nn.Conv2d(3, 128, 5)
        self.conv2 = nn.Conv2d(128, 64, 5)
        self.conv3 = nn.Conv2d(64, 32, 5)
        self.batch_norm = nn.BatchNorm2d(32)
        # self.conv4 = nn.Conv2d(32, 32, 5)

        self.fc1 = nn.Linear(2592, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(self.batch_norm(F.relu(self.conv3(x))))
        # x = self.pool2(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
