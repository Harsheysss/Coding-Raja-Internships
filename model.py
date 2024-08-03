import torch
import torch.nn as nn

class SentimentModel(nn.Module):
    def __init__(self, input_dim):
        super(SentimentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 3)  # 3 classes: negative, neutral, positive

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x