import torch
import torch.nn as nn


# Cut Classifier
class DelayClassPredictor(nn.Module):
    def __init__(self, hidden1_dim, hidden2_dim):
        super(DelayClassPredictor, self).__init__()
        self.fc1 = nn.Linear(32, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        class_output = torch.sigmoid(self.fc3(x))
        return class_output
    

# Delay Predictor
class DelayQoRPredictor(nn.Module):
    def __init__(self, hidden1_dim, hidden2_dim):
        super(DelayQoRPredictor, self).__init__()
        self.fc1 = nn.Linear(32, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        reg_output = self.fc3(x)
        return reg_output
    