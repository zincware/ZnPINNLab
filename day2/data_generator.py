from pyDOE import *
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math 
from scipy import special

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Schrodinger_Boundary(Dataset):
    def __init__(self, num_col_bound = 50):  
        self.num_col_bound = num_col_bound
        num_col_bound_half = math.floor(num_col_bound/2)
        self.t = torch.unsqueeze(torch.tensor(np.squeeze(lhs(1,samples=num_col_bound_half)*math.pi*4/5)).float().to(device), 1)
        self.x = torch.ones_like(self.t).float().to(device)*5
        return
    def __getitem__(self,idx):
        return self.t[idx]
    def __len__(self):
        return len(self.t)
    def getall(self):
        return self.x, self.t
      

class Schrodinger_Initial(Dataset):
    def __init__(self, num_h_init=50, n=0):
        self.num_samples = num_h_init
        factorial = lambda n: 1 if n == 0 else n * factorial(n - 1)
        normfactor = lambda n: (1/torch.sqrt(torch.ones_like(self.x)*factorial(n)*2**n))*(1/torch.pi**(1/4))
        psi = lambda x: normfactor(n)*torch.exp((-x**2) / 2) * special.hermite(n, monic=True)(x.cpu()).to(device)
        self.x = torch.tensor((lhs(1,samples=num_h_init)*10 - 5)).float().to(device)
        self.t = torch.zeros((num_h_init, 1)).float().to(device)
        self.psi = torch.squeeze(torch.stack((
            torch.tensor(psi(self.x)).float().to(device),
            torch.zeros((len(self.x), 1)).to(device)
        ),1))

    def __getitem__(self, idx):
        return self.x[idx], self.t[idx], self.psi[idx]
    
    def __len__(self):
        return len(self.x)
    
    def getall(self):
        return self.x, self.t, self.psi

class Schrodinger(Dataset):
    def __init__(self, num_col_schro = 20000): # returns x,t
        self.num_col_schro = num_col_schro
        self.x = torch.tensor(lhs(1,samples=num_col_schro)*10 - 5).float().to(device)
        self.t = torch.tensor(lhs(1,samples=num_col_schro)*(math.pi*4/5)).float().to(device)
        return 
    def __getitem__(self,idx):
        return self.x[idx], self.t[idx]
    def __len__(self):
        return len(self.x), len(self.t)
    def getall(self):
        return self.x, self.t