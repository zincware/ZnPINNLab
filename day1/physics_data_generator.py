from dataclasses import dataclass

import torch


@dataclass
class PhysicalSystem:
    """
    Class to define the physical system parameters.

    The class is used to define the physical system parameters and the
    initial conditions. The class also contains the function to calculate
    the acceleration of the projectile.

    Attributes
    ----------
    g : float
        Acceleration due to gravity [m/s^2].
    density : float
        Air density [kg/m^3].
    drag_coeff : float
        Drag coefficient.
    cross_area : float
        Cross-sectional area [m^2].
    mass : float
        Mass of the projectile [kg].
    coeff : float
        Coefficient used in the calculation of the acceleration.
    r_0 : torch.tensor
        Initial position.
    v_0 : torch.tensor
        Initial velocity.

    Methods
    -------
    calc_acceleration(t, y)
        Calculates the acceleration of the projectile.
    """

    g: float = 9.81  # Acceleration due to gravity [m/s^2]
    density: float = 1.225  # Air density [kg/m^3]
    drag_coeff: float = 0.47   # Drag coefficient
    cross_area: float = 0.01  # Cross-sectional area [m^2]
    mass: float = 0.1  # Mass of the projectile [kg]
    # Initial Conditions
    r_0: torch.tensor = None  # Initial position
    v_0: torch.tensor = None  # Initial velocity

    def __post_init__(self):
        """
        Post-initialization method.

        The method is used to calculate the coefficient used in the calculation
        of the acceleration.

        Returns
        -------
        Updates the coefficient attribute.
        self.coeff : float
        """
        self.coeff = 0.5 * self.density * self.drag_coeff * self.cross_area
        if self.r_0 is None:
            self.r_0 = torch.tensor([0.0, 0.0])
        if self.v_0 is None:
            self.v_0 = torch.tensor([20.0, 30.0])

    def calc_acceleration(self, t, y):
        """
        Calculates the acceleration of the projectile.

        Parameters
        ----------
        t : torch.tensor
            Current time.
        y : torch.tensor, shape=(2, n)
            Current state of the system, with n being the number of physical dimensions.

        Returns
        -------
        a : torch.tensor
            Acceleration.
        """
        a = torch.zeros_like(y)
        v_abs = torch.norm(y[1])
        a[1] = -self.coeff * v_abs * y[1] / self.mass - torch.tensor([0, self.g])
        a[0] = y[1]
        return a


class ProjectileDataGenerator:
    """
    Class to generate data for projectile motion with air drag.
    """

    def __init__(self, system: PhysicalSystem, step_size: float = 0.001):
        """
        Constructor of the class.

        Parameters
        ----------
        system : PhysicalSystem
            Physical system parameters.
        step_size : float, optional
            Step size for the integration.
        stop_condition : bool
            Condition to stop the integration.
        """
        self.system = system
        self.step_size = step_size

        self.stop_condition = False

        self._y = torch.stack([self.system.r_0, self.system.v_0])
        self._t = torch.tensor([0.0])

        self._y_history = torch.clone(self._y).unsqueeze(0)
        self._t_history = torch.clone(self._t).unsqueeze(0)

        # Final data
        self.time = None
        self.position = None
        self.velocity = None
        self.acceleration = None

    @staticmethod
    def rk4_step(f, t, y, h):
        """
        Runge-Kutta 4th order step.

        Takes a single step using the Runge-Kutta 4th order method.

        Parameters
        ----------
        f : callable
            Function to integrate.
        t : torch.tensor
            Current time.
        y : torch.tensor
            Current state.
        h : float
            Step size.

        Returns
        -------
        t : torch.tensor
            Updated time.
        y : torch.tensor
            Updated state.

        """
        k1 = h * f(t, y)
        k2 = h * f(t + h / 2, y + k1 / 2)
        k3 = h * f(t + h / 2, y + k2 / 2)
        k4 = h * f(t + h, y + k3)
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = t + h
        return t, y

    @staticmethod
    def _update_break_condition(y):
        """
        Update the break condition to stop the integration.

        The condition used to stop the integration is defined as the projectile
        reaching the ground (y = 0).

        Parameters
        ----------
        y : torch.tensor
            Current state.
        """
        if y[0, 1] < 0:
            return True
        else:
            return False

    def integrate(self):
        """
        Integrate the system until the break condition is met.

        The integration is performed using the Runge-Kutta 4th order method.
        With finishing the integration, the final data is generated.

        Returns
        -------
        Update the final data attributes:
        time : torch.tensor
            Time.
        position : torch.tensor
            Position.
        velocity : torch.tensor
            Velocity.
        acceleration : torch.tensor
            Acceleration.
        """

        while self._update_break_condition(self._y) is False:
            self._t, self._y = self.rk4_step(
                self.system.calc_acceleration, self._t, self._y, self.step_size
            )
            self._y_history = torch.cat([self._y_history, self._y.unsqueeze(0)], dim=0)
            self._t_history = torch.cat([self._t_history, self._t.unsqueeze(0)], dim=0)

        self.time = self._t_history.squeeze(1)
        self.position = self._y_history[:, 0, :]
        self.velocity = self._y_history[:, 1, :]
        self.acceleration = torch.vmap(self.system.calc_acceleration, in_dims=0)(
            self._t_history, self._y_history
        )[:, 1, :]
