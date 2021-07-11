import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class flatMSE(nn.ModuleList):
    def __init__(self):
        super().__init__()
        #elf.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss(reduction='none')
        self.beta = 0.999999
    
    def forward(self, x, y):
        #print(x.type())
        #print(y.type())
        
        y = y.type(torch.float)
        flat_x = torch.flatten(x)
        flat_y = torch.flatten(y)
        total_y_1_px = torch.sum(flat_y)
        invert_y = torch.mul(flat_y, -1)
        total_y_0_px = torch.sum(torch.add(invert_y, 1))
        
        reverse_flat_y = torch.add(invert_y, 1)
        
        #print('total 1 pixels')
        #print(total_y_1_px)
        #print('total 0 pixels')
        #print(total_y_0_px)
        effective_1 = torch.div((1-torch.pow(self.beta, total_y_1_px)),1-self.beta)
        effective_0 = torch.div((1-torch.pow(self.beta, total_y_0_px)),1-self.beta)
        averageEffective = 0.5 * torch.add(effective_0,effective_1)

        #print('effective # of 1s')
        #print(effective_1)
        #print('effective # of 0s')
        #print(effective_0)
        weight = torch.mul(torch.add(torch.div(flat_y,effective_1),torch.div(reverse_flat_y, effective_0)), averageEffective)
        #print(flat_y)
        #print(weight)
        #sig_x = self.sigmoid(flat_x)
        loss = self.loss(flat_x, flat_y)
        weighted_loss = torch.mul(weight, loss)
        weighted_loss = torch.mean(weighted_loss)
        return weighted_loss