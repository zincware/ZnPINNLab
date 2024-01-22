import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class AnimationGenerator:
    def __init__(self, train_loss_evolution, test_loss_evolution, predictions_list, train_ds, test_ds):
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
        self.train_loss_evolution = train_loss_evolution
        self.test_loss_evolution = test_loss_evolution
        self.predictions_list = predictions_list
        self.train_ds = train_ds
        self.test_ds = test_ds
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 5))
        self.prediction_plot = None
        self.epoch_text = None
        self.train_loss_plot = None
        self.test_loss_plot = None
    
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
        #self.ax1.set_xlim(0, 37)
        #self.ax1.set_ylim(0, 22)
        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')
        self.ax1.set_title("Model output during training")
        self.prediction_plot = self.ax1.scatter([], [], marker='o', alpha=0.1, c='tab:blue', s=10, label='Predicted output')
        self.ax1.plot(self.test_ds["targets_x"].squeeze(1), self.test_ds["targets_y"].squeeze(1), c='tab:orange', label='True solution')
        self.ax1.plot(self.train_ds["targets_x"].detach().numpy(), self.train_ds["targets_y"].detach().numpy(), 'o', c='g', label='Training Points')
        self.ax1.legend()
        self.epoch_text = self.ax1.text(0.05, 0.95, '', transform=self.ax1.transAxes, ha='left', fontsize=12)

        # Customize the second subplot (right)
        self.ax2.set_xlim(0, len(self.train_loss_evolution))
        self.ax2.set_ylim(
            min(min(self.train_loss_evolution), min(self.test_loss_evolution)),
            max(max(self.train_loss_evolution), max(self.test_loss_evolution))
        )
        self.ax2.set_xlabel('Epochs')
        self.ax2.set_title('Loss evolution during training')
        self.ax2.set_ylabel('Loss')
        self.ax2.set_yscale('log')
        self.train_loss_plot, = self.ax2.plot([], [], label='Train Loss', lw=2, c='r')
        self.test_loss_plot, = self.ax2.plot([], [], label='Test Loss', lw=2, c='g')
        self.ax2.legend(loc='upper right')
        # Adjust layout for subplots
        plt.tight_layout()
    
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
        x_pred, y_pred = self.predictions_list[frame]
        self.prediction_plot.set_offsets(np.column_stack((x_pred.detach().numpy(), y_pred.detach().numpy())))
        epoch_number = (frame) + 1
        self.epoch_text.set_text(f'Epoch: {epoch_number}')
        self.train_loss_plot.set_data(range(frame + 1), self.train_loss_evolution[:frame + 1])
        self.test_loss_plot.set_data(range(frame + 1), self.test_loss_evolution[:frame + 1])
        return self.prediction_plot, self.epoch_text, self.train_loss_plot, self.test_loss_plot
    
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
        # Set the path to the ffmpeg executable
        plt.rcParams['animation.ffmpeg_path'] ='../bin/ffmpeg.exe'
        self._create_plot()
        ani = FuncAnimation(self.fig, self._update, frames=frames, blit=True)
        ani.save(filename, writer='ffmpeg', fps=fps)