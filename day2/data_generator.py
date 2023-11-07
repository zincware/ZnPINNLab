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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Schrodinger_Boundary(Dataset):
    def __init__(self, num_col_bound = 50):  
        self.num_col_bound = num_col_bound
        num_col_bound_half = math.floor(num_col_bound/2)
        self.t = torch.unsqueeze(torch.tensor(np.squeeze(lhs(1,samples=num_col_bound_half)*math.pi/2)).float().to(device),1)
        return
    def __getitem__(self,idx):
        return self.t[idx]
    def __len__(self):
        return len(self.t)
    def getall(self):
        return self.t
    
class Schrodinger_Initial(Dataset):
    def __init__(self, num_h_init = 50):            
        h_func = lambda x : 2 * (1/torch.cosh(x))
        self.num_h_init = num_h_init
        self.x = torch.tensor((lhs(1,samples=num_h_init)*10 - 5)).float().to(device)
        self.t = torch.zeros((num_h_init,1)).float().to(device)
        self.h = torch.squeeze(torch.stack((
            torch.tensor(h_func(self.x)).float().to(device),
            torch.zeros((len(self.x),1)).to(device)
        ),1))
        return
    def __getitem__(self,idx):
        return self.x[idx], self.h[idx]
    def __len__(self):
        return len(self.x)
    def getall(self):
        return self.x, self.t, self.h, 

class Schrodinger(Dataset):
    def __init__(self, num_col_schro = 20000): # returns x,t
        self.num_col_schro = num_col_schro
        x = torch.tensor(lhs(1,samples=num_col_schro)*10 - 5).float().to(device)
        t = torch.tensor(lhs(1,samples=num_col_schro)*(math.pi/2)).float().to(device)
        self.X = torch.squeeze(torch.dstack((x,t))).float()
        return 
    def __getitem__(self,idx):
        return self.X[idx]
    def __len__(self):
        return len(self.X)
    def getall(self):
        return self.X