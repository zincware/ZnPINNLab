import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class AnimationGenerator:
    def __init__(self,predictions_dict, train_ds, test_ds):
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
        self.train_loss_evolution = predictions_dict["train_loss_evolution"]
        self.test_loss_evolution = predictions_dict["test_loss_evolution"]
        self.predictions_list = predictions_dict["predictions_list"]
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
        plt.rcParams['animation.ffmpeg_path'] ='/usr/bin/ffmpeg'
        self._create_plot()
        ani = FuncAnimation(self.fig, self._update, frames=frames, blit=True)
        ani.save(filename, writer='ffmpeg', fps=fps)

class SolutionVisualizer:
    """
    Solution Visualizer Class

    This class provides a convenient way to visualize 
    the solution of the neural network.
    It enables the user to change the title of the left plot 
    and to save the figure as an SVG file.

    parameters
    ----------
    predictions_dict : dict
            A dictionary containing "train_loss_evolution", 
            "test_loss_evolution", "predictions_list", 
            "train_targets_x", "train_targets_y", 
            "test_targets_x", "test_targets_y" 
            and optionally "x_pred_col" and "y_pred_col"
            entries.
    title : str
            The title of the left plot.
    filename : str, optional
            The file name to save the figure as an SVG file. 
            Default is "None".
    
    """


    def __init__(self, predictions_dict):
        self.predictions_dict = predictions_dict

    def plot_solution(self, title, filename=None):
        """
        Plot the solution of the physics-informed neural network.

        This function plots the predicted 
        and true trajectories of the projectile motion
        and the evolution of the training 
        and test loss during the training process.
        It enables the user to change the title of the left plot 
        and to save the figure as an SVG file.

        parameters
        ----------
        title : str
                The title of the left plot.
        filename : str, optional
                The file name to save the figure as an SVG file. 
                Default is "None".
        
        Returns
        -------
        None
        
        """
        # Create a figure with subplots
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # Plot data on the first subplot
        self._customize_first_subplot(axs[0], title)

        # Create an array of epoch numbers for the x-axis
        epochs = np.arange(1, len(self.predictions_dict["train_loss_evolution"]) + 1)

        # Plot loss evolution on the second subplot
        self._customize_second_subplot(axs[1], epochs)

        # Adjust layout for subplots
        plt.tight_layout()

        # Saving SVG if required
        if filename is not None:
            fig.savefig(filename, format='svg', bbox_inches='tight')

        # Show the plot
        plt.show()

    def _customize_first_subplot(self, ax, title):
        """
        Customize the first subplot.

        This function customizes the first subplot by plotting the predicted 
        and true trajectories of the projectile motion.

        parameters
        ----------
        ax : matplotlib.axes.Axes
                The axes object to plot on.
        title : str
                The title of the left plot.
        
        Returns
        -------
        None
        """

        ax.plot(self.predictions_dict["predictions_list"][-1][0].detach().numpy(),
                self.predictions_dict["predictions_list"][-1][1].detach().numpy(),
                'o', alpha=0.01, label='Predicted')

        ax.plot(self.predictions_dict["test_targets_x"],
                self.predictions_dict["test_targets_y"],
                c='tab:orange', label='True Solution')

        ax.plot(self.predictions_dict["train_targets_x"],
                self.predictions_dict["train_targets_y"],
                'o', c='tab:green', label='Training Points')

        if 'x_pred_col' in self.predictions_dict.keys():
            ax.plot(self.predictions_dict["x_pred_col"].detach().numpy(),
                    self.predictions_dict["y_pred_col"].detach().numpy(),
                    '+', c='tab:red', label='Colocation Points')

        ax.legend()

        # Access the legend frame and adjust its alpha value
        legend = ax.legend()
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)  

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(title)

    def _customize_second_subplot(self, ax, epochs):
        """
        Customize the second subplot.

        This function customizes the second subplot 
        by plotting the evolution of the training and test loss.

        parameters
        ----------
        ax : matplotlib.axes.Axes
                The axes object to plot on.
        epochs : numpy.ndarray
                An array of epoch numbers for the x-axis.
        
        Returns
        -------
        None
        """
        ax.plot(epochs,                                         
                self.predictions_dict["train_loss_evolution"],  
                c='r', label='Training loss')

        ax.plot(epochs,                                         
                self.predictions_dict["test_loss_evolution"],   
                c='g', label='Test loss')

        ax.legend()
        ax.set_title('Loss Evolution')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_yscale('log') 