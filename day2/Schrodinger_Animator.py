import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np

class AnimationCreator:
    """
    Animation Creator Class

    This class is designed to create animations of the solution of the Schrödinger equation
    using the neural network model. It enables the User to visualize the solution in different ways:
    - Evolution of the real part prediction during training.
    - Evolution of the imaginary part prediction during training.
    - Cross-section of the final prediction animated along the time axis.
    - Complex Plot of the Solution, to visualize its norm and phase.

    It returns the animations as mp4 files and the last frame as an SVG file.
    """
    def __init__(self, model, period, predictions_list, device='cpu'):
        """
        Constructor for the AnimationCreator class.

        Initializes the model, period and the list of predictions throughout the training.

        Parameters
        ----------
        model : torch.nn.Module
            The physics-informed neural network model.
        period : float
            The period time of the wavefunction.
        predictions_list : list
            A list containing the predictions of the model during training.
        
        Returns
        -------
        None
        """
        # Set the model, period and the list of predictions throughout the training
        self.model = model
        self.period = period
        self.predictions_list = predictions_list
        self.device = device

        # Set the number of x and t values
        self.num_x = 100
        self.num_t = 300
 
    def create_meshgrid(self):
        """
        Create the meshgrid for the position and time tensors.

        This method creates the meshgrid for the position and time tensors
        using the number of x and t values.

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        # Create position and time tensors
        x = torch.linspace(-5, 5, self.num_x)
        self.t = torch.linspace(0, self.period, self.num_t)

        # Create a meshgrid
        self.X, self.T = torch.meshgrid(x, self.t)

    def compute_solution(self):
        """
        Compute the solution of the Schrödinger equation.

        This method computes the solution of the Schrödinger equation
        using the neural network model.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Create a combined input tensor
        _X = torch.tensor(torch.dstack((self.X,self.T)).clone().detach()).float().to(self.device)

        # Compute the predicted real and imaginary parts of the solution
        h_r_hat, h_i_hat = self.model(_X)

        # Detach the tensors from the computational graph
        h_r_hat = h_r_hat.to('cpu').detach()
        h_i_hat = h_i_hat.to('cpu').detach()

        # Combine the real and imaginary parts
        self.h_hat = torch.cat((h_r_hat, h_i_hat),2)

        # Compute the norm of the solution
        h_norm = torch.sqrt((self.h_hat[:,:,0] ** 2) + (self.h_hat[:,:,1] ** 2)).unsqueeze(2)
        self.h_hat = torch.cat((self.h_hat, h_norm),2)

    def create_plot(self):
        """
        Create the plot for the animation.
        
        This method initializes the plot for the animation and sets the axis ticks fontsize.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Plot the norm of the solution
        self.fig = plt.figure(figsize=(15, 15))
        self.ax = plt.axes(projection= '3d')
        
        # Set the tick label font size
        self.ax.xaxis.set_tick_params(labelsize=16)
        self.ax.yaxis.set_tick_params(labelsize=16)
        self.ax.zaxis.set_tick_params(labelsize=16, pad=15)

    def _update_train_real(self, frame):
        """
        Update the plot for the real part prediction during training.

        This method updates the plot for the real part prediction during training
        and sets the axis limits, labels and title.

        Parameters
        ----------
        frame : int
            The current frame of the animation.

        Returns
        -------
        plot_3d_real
            The updated plot for the real part prediction during training.
        """        
        # Clear the axis
        self.ax.clear()

        # Plot the Solution
        self.plot_3d_real = self.ax.plot_wireframe(self.T, self.X, self.predictions_list[frame][:,:,0].detach().numpy(), 
                                                   edgecolor='blue', rstride=5, cstride = 5, label = r"$Ψ_r(x,t)$", linewidth = 1),

        # Set the view angle
        self.ax.view_init(17, -19)

        # Set the axis limits
        self.ax.set_xlim(0, self.period)
        self.ax.set_zlim(-1, 1)

        # Set the axis labels
        self.ax.set_ylabel(r"x$\left[\sqrt{\frac{\hbar}{m\omega}}\right]$", fontsize=16, labelpad=10)
        self.ax.set_xlabel(r"$t[\omega^{-1}]$", fontsize=16, labelpad=10)
        self.ax.set_zlabel(r"$Ψ_r(x,t)$", fontsize=16, labelpad=30)

        # Set the title and the legend
        self.ax.set_title('Evolution of the real part prediction during Training', fontsize=20)
        self.ax.legend(fontsize=16)

        # Save the last frame as an SVG file
        if frame == len(self.predictions_list)-1:
            self.fig.savefig(f'train_evolution_real.svg', format='svg', bbox_inches='tight')

        return self.plot_3d_real
    
    def _update_train_imag(self, frame):
        """
        Update the plot for the imaginary part prediction during training.

        This method updates the plot for the imaginary part prediction during training
        and sets the axis limits, labels and title.

        Parameters
        ----------
        frame : int
            The current frame of the animation.

        Returns
        -------
        plot_3d_real
            The updated plot for the imaginary part prediction during training.
        """  
        self.ax.clear()

        # Plot the Solution
        self.plot_3d_imag = self.ax.plot_wireframe(self.T, self.X, self.predictions_list[frame][:,:,1].detach().numpy(), 
                                                   edgecolor='red', rstride=5, cstride = 5, linewidth=1, label = r"$Ψ_i(x,t)$")
        
        # Set the view angle
        self.ax.view_init(17, -19)

        # Set the axis limits
        self.ax.set_xlim(0, self.period)
        self.ax.set_zlim(-1, 1)

        # Set the axis labels
        self.ax.set_ylabel(r"x$\left[\sqrt{\frac{\hbar}{m\omega}}\right]$", fontsize=16, labelpad=10)
        self.ax.set_xlabel(r"$t[\omega^{-1}]$", fontsize=16, labelpad=10)
        self.ax.set_zlabel(r"$Ψ_i(x,t)$", fontsize=16, labelpad=30)
        
        # Set the title and the legend
        self.ax.set_title('Evolution of the imaginary part prediction during Training', fontsize=20)
        self.ax.legend(fontsize=16)

        # Save the last frame as an SVG file
        if frame == len(self.predictions_list)-1:
            self.fig.savefig(f'train_evolution_imag.svg', format='svg', bbox_inches='tight')

        return self.plot_3d_imag

    def _update_cross(self, frame):
        """
        Update the plot for the cross-section of the final prediction animated along the time axis.

        This method updates the plot for the cross-section of the final prediction animated along the time axis
        and sets the axis limits, labels and title.

        Parameters
        ----------
        frame : int
            The current frame of the animation.
        
        Returns
        -------
        plot_3d_real
            The updated wireframe plot for the cross-section of the real part prediction.
        plot_2d_real
            The updated line plot for the cross-section of the real part prediction.
        plot_3d_imag
            The updated wireframe plot for the cross-section of the imaginary part prediction.
        plot_2d_imag
            The updated line plot for the cross-section of the imaginary part prediction.
        """
        self.ax.clear()

        # Set the (time) range of the indices to be displayed
        t_range_indices = torch.arange(0,frame+1)
        
        # Cut the solution and the meshgrid
        self.h_hat_cut = self.h_hat[:, t_range_indices, :]
        self.X_cut = self.X[:, t_range_indices]
        self.T_cut = self.T[:, t_range_indices]

        # Plot the cutted solutions
        self.plot_3d_real = self.ax.plot_wireframe(self.T_cut, self.X_cut, self.h_hat_cut[:,:,0].detach().numpy(), cmap='PuBu', edgecolor='blue', rstride=5, cstride = 5, linewidth=0.2),
        self.plot_3d_imag = self.ax.plot_wireframe(self.T_cut, self.X_cut, self.h_hat_cut[:,:,1].detach().numpy(), cmap='OrRd', edgecolor='red', rstride=5, cstride = 5, linewidth=0.2)
        
        # Plot the 2D cross-sections
        self.plot_2d_real = self.ax.plot( self.T[:, frame].numpy(), self.X[:,frame].numpy(), self.h_hat[:,frame,0].detach().numpy(), zdir='z', color='blue', label=r"$Ψ_r(x,t)$")
        self.plot_2d_imag = self.ax.plot( self.T[:, frame].numpy(), self.X[:,frame].numpy(), self.h_hat[:,frame,1].detach().numpy(), zdir='z', color='red', label=r"$Ψ_i(x,t)$")

        # Set the view angle
        self.ax.view_init(17, -19)

        # Set the axis limits
        self.ax.set_xlim(0, self.period)
        self.ax.set_zlim(-1, 1)

        # Set the axis labels
        self.ax.set_ylabel(r"x$\left[\sqrt{\frac{\hbar}{m\omega}}\right]$", fontsize=16, labelpad=10)
        self.ax.set_xlabel(r"$t[\omega^{-1}]$", fontsize=16, labelpad=10)
        self.ax.set_zlabel(r"$Ψ(x,t)$", fontsize=16, labelpad=30)

        # Set the title and the legend
        self.ax.set_title('Cross-section evolution along the time axis', fontsize=20)
        self.ax.legend(fontsize=16)

        # Save the last frame as an SVG file
        if frame == len(self.t)-1:
            self.fig.savefig(f'Cross-section.svg', format='svg', bbox_inches='tight')

        return self.plot_3d_real, self.plot_2d_real, self.plot_3d_imag#, self.plot_2d_imag

    def _update_complex_plot(self, frame):
        """
        Update the plot for the complex plot of the solution.

        This method updates the plot for the complex plot of the solution
        and sets the axis limits, labels and title.

        Parameters
        ----------
        frame : int
            The current frame of the animation.
        
        Returns
        -------
        plot_2d_real
            The updated 2D plot for the real part of the solution.
        plot_2d_imag
            The updated 2D plot for the imaginary part of the solution.
        plot_line
            The updated line plot for the cross-section of the solution.
        plot_wave
            The updated 3D plot for the solution.
        plot_line_real
            The updated line plot for the real part of the solution.
        plot_line_imag 
            The updated line plot for the imaginary part of the solution.
        """
        self.ax.clear()
        
        self.plot_wave = self.ax.plot(self.h_hat[:,frame,1].detach().numpy(), self.X[:,frame].numpy(), self.h_hat[:,frame,0].detach().numpy(), color='black', label='|Ψ(x,t)|')
        self.plot_2d_real = self.ax.plot( self.X[:,frame].numpy(), self.h_hat[:,frame,0].detach().numpy(),zs=0, zdir='x', color='blue', label=r"$Ψ_r(x,t)$")
        self.plot_2d_imag = self.ax.plot( self.h_hat[:,frame,1].detach().numpy(), self.X[:,frame].numpy(), zs=0, zdir='z', color='red', label=r"$Ψ_i(x,t)$")
        self.plot_line = self.ax.plot(self.h_hat[int(self.num_x/2),:frame+1,1].detach().numpy(), self.h_hat[int(self.num_x/2),:frame+1,0].detach().numpy(), 
                                      zs=0, zdir='y',linestyle='--', color='black')

        line_y_real = np.array([0, self.h_hat[int(self.num_x/2), frame, 1].detach().numpy()])
        line_z_real = np.array([self.h_hat[int(self.num_x/2), frame, 0].detach(), self.h_hat[int(self.num_x/2), frame, 0].detach()])
        
        line_y_imag = np.array([0, self.h_hat[int(self.num_x/2), frame, 0].detach().numpy()])
        line_z_imag = np.array([self.h_hat[int(self.num_x/2), frame, 1].detach(), self.h_hat[int(self.num_x/2), frame, 1].detach()])

        self.plot_line_real = self.ax.plot(line_y_real, line_z_real, zs=0, zdir='y', linestyle='-.', color='blue')
        self.plot_line_imag = self.ax.plot( line_z_imag, line_y_imag, zs=0, zdir='y', linestyle='-.', color='red')

        # Set the aspect ratio of the plot
        self.ax.set_box_aspect([1, 1, 1])

        # Set the view angle
        self.ax.view_init(20, -80+frame/15)

        # Set the axis limits
        self.ax.set_ylim(-5, 5)
        self.ax.set_xlim(-1, 1)
        self.ax.set_zlim(-1, 1)

        # Set the axis labels
        self.ax.set_ylabel(r"x$\left[\sqrt{\frac{\hbar}{m\omega}}\right]$", labelpad=10, fontsize=16)
        self.ax.set_xlabel(r"$Ψ_i(x,t)$", fontsize=16, labelpad=10)
        self.ax.set_zlabel(r"$Ψ_r(x,t)$", fontsize=16, labelpad=25)

        # Set the title and the legend
        self.ax.set_title('Complex plot of the Wavefunction', fontsize=20)
        self.ax.legend(fontsize=16)

        # Save the last frame as an SVG file
        if frame == len(self.t)-1:
            self.fig.savefig(f'Complex_plot.svg', format='svg', bbox_inches='tight')
        

        return  self.plot_2d_real, self.plot_2d_imag, self.plot_line, self.plot_wave, self.plot_line_real, self.plot_line_imag
    
    def create_train_evolution_real_animation(self, fps=30):
        """
        Create the animation for the evolution of the real part prediction during training.

        This method creates the animation for the evolution of the real part prediction during training
        and saves it to a file using the specified filename, writer, and fps.

        Parameters
        ----------
        filename : str
            The filename of the animation.
        fps : int, optional
            The frames per second of the animation. Default is 30.
        
        Returns
        -------
        None
        """
        # Set the path to the ffmpeg executable
        plt.rcParams['animation.ffmpeg_path'] ='/usr/bin/ffmpeg'
        
        # Create the plot for the animation
        self.create_plot()
        
        # Create the animation using FuncAnimation
        ani = FuncAnimation(self.fig, self._update_train_real, frames=len(self.predictions_list), blit=False)
        
        # Save the animation to a file using the specified filename, writer, and fps
        ani.save('train_evolution_real.mp4', writer='ffmpeg', fps=fps)

    def create_train_evolution_imag_animation(self, fps=30):
        """
        Create the animation for the evolution of the imaginary part prediction during training.
        
        This method creates the animation for the evolution of the imaginary part prediction during training
        and saves it to a file using the specified filename, writer, and fps.

        Parameters
        ----------
        filename : str
            The filename of the animation.
        fps : int, optional
            The frames per second of the animation. Default is 30.

        Returns
        -------
        None
        """ 
        # Set the path to the ffmpeg executable
        plt.rcParams['animation.ffmpeg_path'] ='/usr/bin/ffmpeg'
        
        # Create the plot for the animation
        self.create_plot()
        
        # Create the animation using FuncAnimation
        ani = FuncAnimation(self.fig, self._update_train_imag, frames=len(self.predictions_list), blit=False)
        
        # Save the animation to a file using the specified filename, writer, and fps
        ani.save('train_evolution_imag.mp4', writer='ffmpeg', fps=fps)

    def create_cross_section_animation(self, fps=30):
        """
        Create the animation for the cross-section of the final prediction animated along the time axis.

        This method creates the animation for the cross-section of the final prediction animated along the time axis
        and saves it to a file using the specified filename, writer, and fps.

        Parameters
        ----------
        filename : str
            The filename of the animation.
        fps : int, optional

        Returns
        -------
        None
        """
        # Set the path to the ffmpeg executable
        plt.rcParams['animation.ffmpeg_path'] ='/usr/bin/ffmpeg'
        
        # Create the plot for the animation
        self.create_plot()
        
        # Create the animation using FuncAnimation
        ani = FuncAnimation(self.fig, self._update_cross, frames=300, blit=False)
        
        # Save the animation to a file using the specified filename, writer, and fps
        ani.save('cross_section.mp4', writer='ffmpeg', fps=fps)
    
    def create_complex_plot_animation(self, fps=30):
        """
        Create the animation for the complex plot of the solution.

        This method creates the animation for the complex plot of the solution
        and saves it to a file using the specified filename, writer, and fps.

        Parameters
        ----------
        filename : str
            The filename of the animation.
        fps : int, optional
            The frames per second of the animation. Default is 30.

        Returns
        -------
        None
        """
        # Set the path to the ffmpeg executable
        plt.rcParams['animation.ffmpeg_path'] ='/usr/bin/ffmpeg'
        
        # Create the plot for the animation
        self.create_plot()
        
        # Create the animation using FuncAnimation
        ani = FuncAnimation(self.fig, self._update_complex_plot, frames=300, blit=False)
        
        # Save the animation to a file using the specified filename, writer, and fps
        ani.save('complex_plot.mp4', writer='ffmpeg', fps=fps)
    
