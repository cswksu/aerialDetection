import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class diceLossTwoClass(nn.ModuleList):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, y):
        x = self.softmax(x)
        flat_x_pos_prob = torch.flatten(x[:, 1, :, :])
        flat_y = torch.flatten(y).float()
        intersection = torch.dot(flat_y, flat_x_pos_prob)
        total = torch.sum(flat_x_pos_prob+flat_y)
        return 1 - (intersection+1)/(total + 1)