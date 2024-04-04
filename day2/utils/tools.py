import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

class SolutionVisualizer:
    """
    Visualize the solution of the Schrödinger equation.

    This class visualizes the solution of the Schrödinger equation by using the
    neural network model.
    It enables the User to visualize the solution in different ways:
    - Contour plot of the norm of the solution.
    - Contour plot of the real part of the solution.
    - Contour plot of the imaginary part of the solution.
    - 3D surface plot of the solution.
    - Cross-section of the solution at t=period/2.

    The Contour plots can be saved as SVG files by setting 'save_svg' to True.
    The Name of the SVG files can be customized by setting 'n' and 'potential_scaling' to the desired value.

    Parameters
    ----------
    model : torch.nn.Module
        The physics-informed neural network model.
    period : float
        The period time of the wavefunction.
    potential_scaling : bool
        Whether the potential scaling is set to True of false.
    n : int
        The n-th solution of the Schrodinger equation.
    save_svg : bool, optional
        Whether to save the contour plots as SVG files. Default is False.
    
    Returns
    -------
    None
       
    """
    def __init__(self, model : nn.Module, period : float, potential_scaling : bool, n : int, save_svg : bool):
        self.model = model
        self.period = period
        self.potential_scaling = potential_scaling
        self.n = n
        self.save_svg = save_svg

    def visualize(self):
        result = "True" if self.potential_scaling else "False"
        #%matplotlib inline
        # Create position and time arrays
        x = np.linspace(-5, 5, 100)
        t = np.linspace(0, self.period, 300)

        # Create a meshgrid
        X, T = np.meshgrid(x, t)

        # Create a combined input tensor
        _X = torch.tensor(np.dstack((X, T))).float()

        # Compute the predicted real and imaginary parts of the solution
        with torch.no_grad():
            h_r_hat, h_i_hat = self.model(_X)

        # Combine the real and imaginary parts
        h_hat = torch.cat((h_r_hat, h_i_hat), 2)

        # Compute the norm of the solution
        h_norm = np.sqrt((h_hat[:, :, 0] ** 2) + (h_hat[:, :, 1] ** 2))

        # Plot the norm of the solution
        fig = plt.figure(figsize=(16, 2))
        ax = plt.axes()

        cf = ax.contourf(T, X, h_norm, 30)
        plt.colorbar(cf)
        plt.title(r"$|Ψ(x,t)|$")
        plt.ylabel(r"x$\left[\sqrt{\frac{\hbar}{m\omega}}\right]$")
        plt.xlabel(r"$t[\omega^{-1}]$")
        plt.show()
        # Save the figure as an SVG file
        if self.save_svg:
            fig.savefig(f'Contour_norm_{self.n}_{result}.svg', format='svg', bbox_inches='tight')

        # Plot the real part of the solution
        fig = plt.figure(figsize=(16, 2))
        ax = plt.axes()

        cf = ax.contourf(T, X, h_hat[:, :, 0], 60, cmap='RdBu')
        plt.colorbar(cf)
        plt.title(r"$Ψ_r(x,t)$")
        plt.ylabel(r"x$\left[\sqrt{\frac{\hbar}{m\omega}}\right]$")
        plt.xlabel(r"$t[\omega^{-1}]$")
        plt.show()
        # Save the figure as an SVG file
        if self.save_svg:
            fig.savefig(f'Contour_Real_{self.n}_{result}.svg', format='svg', bbox_inches='tight')

        # Plot the imaginary part of the solution
        fig = plt.figure(figsize=(16, 2))
        ax = plt.axes()

        cf = ax.contourf(T, X, h_hat[:, :, 1], 60, cmap='RdBu')
        plt.colorbar(cf)
        plt.title(r"$Ψ_i(x,t)$")
        plt.ylabel(r"x$\left[\sqrt{\frac{\hbar}{m\omega}}\right]$")
        plt.xlabel(r"$t[\omega^{-1}]$")
        plt.show()
        # Save the figure as an SVG file
        if self.save_svg:
            fig.savefig(f'Contour_Imag_{self.n}_{result}.svg', format='svg', bbox_inches='tight')

        # Plot the 3D surface of the solution
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlabel(r"$t[\omega^{-1}]$")
        ax.set_ylabel(r"x$\left[\sqrt{\frac{\hbar}{m\omega}}\right]$")
        ax.set_zlabel(r"$|Ψ(x,t)|$")
        ax.plot_wireframe(T, X, h_norm)
        ax.set_title('3D Surface of the norm of the solution')
        plt.show()

        # Plot a cross-section of the solution at t=period/2
        x_slice = x[:]
        h_slice = h_norm[150, :]
        plt.plot(x_slice, h_slice)
        plt.title('Solution Cross-section at t=period/2')
        plt.xlabel(r"x$\left[\sqrt{\frac{\hbar}{m\omega}}\right]$")
        plt.ylabel(r"$|Ψ(x,t)|$")
        plt.show()

def logarithmic_sampling(num_epochs):
    """
    This function performs logarithmic sampling of the number of epochs.
    It is used to sample the number of epochs for the training procedure.
    The logarithmic sampling is performed using the following steps:

    Parameters
    ----------
    num_epochs : int
        The number of epochs for training.
    
    Returns
    -------
    List[int]
        A list containing the logarithmically sampled epochs.
    """
    min_val = 1.0
    max_val = num_epochs
    exponent = 6.0  # Sharpness
    pred_epoch_list = []

    for i in np.arange(0.0, 1+(1/543), 1/543):
        result = int(pow(i, exponent) * (max_val - min_val) + min_val)
        if result not in pred_epoch_list:
            pred_epoch_list.append(result)

    return pred_epoch_list

def get_prediction(model, period, device):
    """
    Get the predicted solution of the Schrödinger equation.

    This function computes the predicted solution of the Schrödinger equation
    using the trained PINN model. The following steps are involved:

    Step 1: Position and Time Array Creation
        - Create position and time arrays using the 'torch.linspace' method.
    
    Step 2: Meshgrid Creation  
        - Create a meshgrid using the 'torch.meshgrid' method.
        - Concatenate the position and time arrays along the 2 axis.

    Step 3: Model Prediction   
        - Use the trained model to predict the real and imaginary parts of the solution.
        - Compute the norm of the predicted solution.

    Parameters  
    ----------
    model : torch.nn.Module
        The trained physics-informed neural network model.
    period : float
        The period of the solution.
    device : str
        The device to run the model on.

    Returns
    -------
    torch.Tensor
        The predicted real and imaginary parts of the solution.
    """
    # Create position and time arrays
    x = torch.linspace(-5,5,100)
    t = torch.linspace(0,period,300)

    # Create a meshgrid
    X,T = torch.meshgrid(x,t)
    _X = torch.tensor(torch.dstack((X,T))).clone().detach().float().to(device)

    # Compute the predicted real and imaginary parts of the solution
    h_r_hat, h_i_hat = model(_X)
    h_hat = torch.cat((h_r_hat, h_i_hat),2)
    h_norm = torch.sqrt((h_hat[:,:,0] ** 2) + (h_hat[:,:,1] ** 2)).unsqueeze(2)
    h_hat = torch.cat((h_hat, h_norm), 2)
    return h_hat

def visualize_training(train_loss_evolution, test_loss_evolution):
    """
    Visualize the evolution of the loss during training.

    This function plots the evolution of the loss during training.
    The following steps are involved:


    Parameters
    ----------
    train_loss_evolution : list
        A list containing train loss values during training.
    test_loss_evolution : list
        A list containing test loss values during training.

    Returns
    -------
    None
    """
    # Step 1: Create a figure and axis
    fig, ax = plt.subplots()

    # Add labels and a legend
    ax.plot(np.linspace(0, len(train_loss_evolution), len(train_loss_evolution)), train_loss_evolution, c='r', label='Training loss')
    ax.plot(np.linspace(0, len(test_loss_evolution), len(test_loss_evolution)), test_loss_evolution, c='g', label='Test loss')
    ax.legend()
    ax.set_title('Loss Evolution')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')  # Set y-axis to logarithmic scale

    # Show the plot
    plt.show()

def visualize_imported_data(boundary_vals, initial_vals, schrodinger_vals, shared_data, save_svg=False):
    """
    This function visualizes the imported data from the data_generator.py file.
    The imported data are the boundary, initial and schrodinger data.
    It enables the user to save the figure as a svg file.
    Therefore, set the save_svg parameter to True.

    Parameters
    ----------
    boundary_vals : torch.tensor
        boundary data
    initial_vals : torch.tensor
        initial data
    schrodinger_vals : torch.tensor
        schrodinger data
    shared_data : SharedData
        shared data
    save_svg : bool, optional
        save figure as svg file, by default False
    
    Returns
    -------
    None
        None
    """
    n = shared_data.n
    period = shared_data.period
    schr = schrodinger_vals.getall()
    init = initial_vals.getall()
    bound = boundary_vals.getall()

    schrodinger_x = schr[0].cpu()
    schrodinger_t = schr[1].cpu()

    boundary_x = bound[0].cpu()
    boundary_t = bound[1].cpu()

    init_x = init[0].cpu()
    init_t = init[1].cpu()

    x, t, psi = init

    
    #%matplotlib inline
    # Create a 3D plot
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection= '3d')

    # Plotting in 3D
    ax.scatter(init_t, init_x, psi[:, 0].cpu(), marker='x', label='Initial data real part')
    ax.scatter(init_t, init_x, psi[:, 1].cpu(), marker='x', c='r', label='Initial data imaginary part')
    ax.scatter(boundary_t, boundary_x, zs=0, marker='x', c='g', label='Boundary data')
    ax.scatter(boundary_t, -boundary_x, zs=0, marker='x', c='g')
    schrodinger_plot = ax.scatter(schrodinger_t, schrodinger_x, zs=0, marker='x', alpha=1, label='Schrodinger data')
    
    # Set labels and title
    ax.set_title(fr'Training data for quantum number $n$ = {n}', fontsize=16)
    ax.set_ylabel(r"$x\,\left[\sqrt{\frac{\hbar}{m\omega}}\right]$",)# fontsize=16)
    ax.set_xlabel(r"$t\,[\omega^{-1}]$",)# fontsize=16)
    ax.tick_params(axis='both',)# labelsize=14)
    ax.set_zlabel(rf"$Ψ_{n}(x,t)$",)# fontsize=16)
    ax.set_zticks([ 0, 1])
    ax.set_yticks([-5, 0, 5])
    ax.set_xticks([0, period/2, period])
    ax.set_xticklabels(['$0$', f'${period/(2*math.pi)}\pi$', f'${period/math.pi}\pi$'])

    # Plotting Training Data
    ax.set_xlabel(r"$t\,[\omega^{-1}]$", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='upper right', fontsize=16)
    schrodinger_plot.set_alpha(0.1) # Set alpha to 0.1 for better visibility
    plt.show()

    # Saving SVG if required
    if save_svg:
        fig.savefig(f'Trainingdata_n={n}.svg', format='svg', bbox_inches='tight')

    # Plotting Initial Condition
    fig, ax = plt.subplots()
    ax.plot(x.cpu(), psi[:, 0].cpu(), 'x', label='Real part')
    ax.plot(x.cpu(), psi[:, 1].cpu(), 'x', c='r', label='Imaginary part')
    ax.set_title("Initial condition", fontsize=20)
    ax.set_xlabel(r"$x\,\left[\sqrt{\frac{\hbar}{m\omega}}\right]$", fontsize=16)
    ax.set_ylabel(fr"$Ψ_{n}(x,t=0)$", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=14, loc='upper left')
    plt.show()

    # Saving SVG if required
    if save_svg:
        fig.savefig(f'Initial_Condition_n={n}.svg', format='svg', bbox_inches='tight')

    pass
