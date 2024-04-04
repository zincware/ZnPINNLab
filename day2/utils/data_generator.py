from pyDOE import *
import torch
from torch.utils.data import Dataset
import numpy as np
import math
from scipy import special

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SharedData:
    """
    Shared Data Class
    
    This class defines the data that is shared between the different datasets.

    Attributes
    ----------
    num_col_bound : int
        Number of boundary data points.
    num_h_init : int
        Number of initial data points.
    num_col_schro : int
        Number of Schrödinger equation data points.
    n : int
        Quantum number.
    """
    def __init__(self, num_col_bound=50, num_h_init=50, num_col_schro=20000, n=0):
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Generate common values
        self.num_col_bound = num_col_bound
        self.num_h_init = num_h_init
        self.num_col_schro = num_col_schro
        self.n = n
        self.period = 4*math.pi/(1+2*self.n)

        num_col_bound_half = math.floor(num_col_bound/2)
        self.t_bound = torch.unsqueeze(torch.tensor(np.squeeze(lhs(1, samples=num_col_bound_half)*4*math.pi/(1+2*self.n))).float().to(device), 1)

        self.x_init = torch.tensor((lhs(1, samples=num_h_init)*10 - 5)).float().to(device)
        self.t_init = torch.zeros((num_h_init, 1)).float().to(device)

        self.x_schro = torch.tensor(lhs(1, samples=num_col_schro)*10 - 5).float().to(device)
        self.t_schro = torch.tensor(lhs(1, samples=num_col_schro)*(4*math.pi/(1+2*self.n))).float().to(device)



class Schrodinger_Boundary(Dataset):
    """
    Schrodinger Boundary Dataset Class

    This class defines the boundary dataset for the Schrödinger equation.

    Attributes
    ----------
    shared_data : class
        Shared data between the datasets.

    Methods
    -------
    __init__(shared_data)
        Initializes the dataset.
    __getitem__(idx)
        Returns the item at the specified index.
    __len__()
        Returns the length of the dataset.
    getall()
        Returns all items in the dataset.
    """
    def __init__(self, shared_data):
        self.t = shared_data.t_bound
        self.x = torch.ones_like(self.t).float().to(device)*5
        return

    def __getitem__(self, idx):
        return self.t[idx]

    def __len__(self):
        return len(self.t)

    def getall(self):
        return self.x, self.t


class Schrodinger_Initial(Dataset):
    """
    Schrodinger Initial Dataset Class

    This class defines the initial dataset for the Schrödinger equation.

    Attributes
    ----------
    shared_data : class
        Shared data between the datasets.
    
    Methods
    -------
    __init__(shared_data)
        Initializes the dataset.
    __getitem__(idx)
        Returns the item at the specified index.
    __len__()
        Returns the length of the dataset.
    getall()
        Returns all items in the dataset.
    """
    def __init__(self, shared_data):
        self.n = shared_data.n
        self.x = shared_data.x_init
        self.t = shared_data.t_init
        factorial = lambda n: 1 if n == 0 else n * factorial(n - 1)
        normfactor = lambda n: (1/torch.sqrt(torch.ones_like(self.x)*factorial(n)*2**n))*(1/torch.pi**(1/4))
        psi = lambda x: normfactor(self.n)*torch.exp((-x**2) / 2) * special.hermite(self.n, monic=True)(x.cpu()).to(device)
        self.psi = torch.squeeze(torch.stack((
            torch.tensor(psi(self.x)).float().to(device),
            torch.zeros((len(self.x), 1)).to(device)
        ), 1))

    def __getitem__(self, idx):
        return self.x[idx], self.t[idx], self.psi[idx]

    def __len__(self):
        return len(self.x)

    def getall(self):
        return self.x, self.t, self.psi

class Schrodinger(Dataset):
    """
    Schrodinger Dataset Class

    This class defines the dataset for the Schrödinger equation.

    Attributes
    ----------
    shared_data : class
        Shared data between the datasets.
    
    Methods
    -------
    __init__(shared_data)
        Initializes the dataset.
    __getitem__(idx)
        Returns the item at the specified index.
    __len__()
        Returns the length of the dataset.
    getall()
        Returns all items in the dataset.
    """
    def __init__(self, shared_data):
        self.x = shared_data.x_schro
        self.t = shared_data.t_schro
        return

    def __getitem__(self, idx):
        return self.x[idx], self.t[idx]

    def __len__(self):
        return len(self.x)

    def getall(self):
        return self.x, self.t