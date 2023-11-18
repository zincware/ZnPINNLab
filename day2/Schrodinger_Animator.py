import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

class AnimationGenerator:
    def __init__(self,predcitions_list, predictions_list_r, predictions_list_i, psi_min, psi_max):
        """
        Initializes an AnimationGenerator object.

        Parameters:
        -----------
        train_loss_evolution : list
            A list of floats representing the training loss at each epoch.
        test_loss_evolution : list
            A list of floats representing the test loss at each epoch.
        predictions_list : list
            A list of tuples representing the predicted output at each epoch.
        train_ds : dict
            A dictionary containing the training dataset.
        test_ds : dict
            A dictionary containing the training dataset of the true solution.

        Returns:
        --------
        None
        """
        self.psi_min = psi_min
        self.psi_max = psi_max

        self.predcitions_list = predcitions_list
        self.predictions_list_r = predictions_list_r
        self.predictions_list_i = predictions_list_i
        self.X = torch.linspace(-5,5,100).reshape(-1,1)
        
        self.fig, self.ax1 = plt.figure(), plt.axes()
        self.prediction_plot = None
        self.prediction_plot_r = None
        self.prediction_plot_i = None
    
    def _create_plot(self):
        """
        Creates the plot for the animation.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        # Customize the first subplot (left)
        
        self.ax1.set_xlim(-5, 5)
        self.ax1.set_ylim(1.1*self.psi_min, 1.1*self.psi_max)
        self.ax1.set_xlabel('Position x')
        self.ax1.set_ylabel('Wavefunction(x)')
        self.ax1.set_title("Model output over time")
        self.prediction_plot, = plt.plot([], [], alpha=1, c='black', label='|psi|^2')
        self.prediction_plot_r, = plt.plot([], [], alpha=1, c='tab:blue', label='Real part')
        self.prediction_plot_i, = plt.plot([], [], alpha=1, c='tab:red', label='Imaginary part')
        self.ax1.legend(loc='upper right')
        #self.epoch_text = self.ax1.text(0.05, 0.95, '', transform=self.ax1.transAxes, ha='left', fontsize=12)

    
    def _update(self, frame):
        """
        Updates the plot for each frame of the animation.

        Parameters:
        -----------
        frame : int
            The current frame number.

        Returns:
        --------
        tuple
            A tuple containing the updated scatter plot, epoch text, and loss lines.
        """
        hf_r, hf_i, h = self.predictions_list_r[frame], self.predictions_list_i[frame], self.predcitions_list[frame]
         # Update the line plot for the both parts
        self.prediction_plot.set_data(self.X.detach().numpy(), h.detach().numpy())
        self.prediction_plot_r.set_data(self.X.detach().numpy(), hf_r.detach().numpy())
        self.prediction_plot_i.set_data(self.X.detach().numpy(), hf_i.detach().numpy())

        return self.prediction_plot_r, self.prediction_plot_i, self.prediction_plot

    def create_animation(self, filename, frames, fps=30):
        """
        Creates the animation and saves it to a file.

        Parameters:
        -----------
        filename : str
            The name of the file to save the animation to.
        frames : int, optional
            The number of frames in the animation. Defaults to the length of the train loss evolution.
        fps : int, optional
            The frames per second of the animation. Defaults to 30.

        Returns:
        --------
        None
        """
        plt.rcParams['animation.ffmpeg_path'] ='./bin/ffmpeg.exe'
        self._create_plot()
        ani = FuncAnimation(self.fig, self._update, frames=frames, blit=True)
        ani.save(filename, writer='ffmpeg', fps=fps)