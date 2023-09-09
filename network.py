import math
import torch
import torch.nn as nn 
import pandas as pd
import numpy as np

device = 'cpu'

class Hard(torch.autograd.Function):
    """
    Hard Mask를 위한 사용자 정의 Step Function.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.round(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output
  

class Mask(nn.Module):
    """
    Mask 분포
    """
    def __init__(self, dim):
        super().__init__()
        self.hard = Hard.apply
        
        self.eps = 1.0
        self.sig = 2.0
        self.noise = torch.randn(dim)
        self.mu = torch.tensor([0.5] * dim)
        self.mu = nn.Parameter(self.mu)

    def sample(self, noisy=True): 
        noise = self.sig * self.noise.normal_()
        noise = noise * self.eps
        mask = self.mu + noise*noisy
        mask = torch.softmax(mask, 0)
        mask = torch.clamp(mask, 0.0, 1.0)
        return mask

    def hard(self, m):
        return self.hard(m).float()

    def cost(self, x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) 


class Rnet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer1 = nn.Linear(dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)
        self.act = nn.ReLU()

    def forward(self, w):
        x = self.layer1(w)
        x = self.act(x)
        x = self.layer2(x)
        x = self.act(x)
        x = self.layer3(x)
        return x
    