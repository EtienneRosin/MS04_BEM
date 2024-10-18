
from helmoltz_2d_BEM.geometry.obstacles import Obstacle
from helmoltz_2d_BEM.utils import MSColors

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Ellipse(Obstacle):
    """
    Class to represent an elliptical obstacle.

    Attributes
    ----------
    N_e : int
        number of discretized elements
    a : float
        semi-major axis of the ellipse
    b : float
        semi-minor axis of the ellipse
    center : list|np.ndarray
        center of the ellipse
    y_e: np.ndarray
        nodes of the discretization
    Gamma_e: np.ndarray
        element indices
    a_e: np.ndarray
        element's first node
    b_e: np.ndarray
        element's last node
    y_e_m: np.ndarray
        element's middle point
    y_e_d: np.ndarray
        element's difference vector
    """
    def __init__(self, N_e: int, a: float, b: float, center: list|np.ndarray = None) -> None:
        """
        Constructs the ellipse object.

        Parameters
        ----------
            N_e : int
                number of elements for the boundary discretization
            a : float
                semi-major axis of the ellipse
            b : float
                semi-minor axis of the ellipse
            center : list|np.ndarray
                center of the ellipse
        """
        self.N_e = self._validate_N_e(N_e)
        self.a = self._validate_axis(a)
        self.b = self._validate_axis(b)
        self.center = self._validate_center(center)
        
        self.y_e, self.Gamma_e = self._construct_mesh()
        self.a_e = self.y_e[:-1]
        self.b_e = self.y_e[1:]
        
        self.y_e_m = np.mean(self.y_e[self.Gamma_e], axis=1)
        self.y_e_d = 0.5 * (self.b_e - self.a_e)

    def _validate_N_e(self, N_e: int) -> int:
        """
        @brief Validate the N_e input.

        @param N_e: number of elements for the discretization.
        @return: validated number of elements.
        """
        if N_e < 4:
            raise ValueError("N_e should be >= 4.")
        return N_e

    def _validate_axis(self, axis: float) -> float:
        """
        @brief Validate the axis input.

        @param axis: semi-major or semi-minor axis.
        @return: validated axis length.
        """
        if axis <= 0:
            raise ValueError("Axis length should be > 0.")
        return axis
    
    def _validate_center(self, center: list|np.ndarray) -> np.ndarray:
        """
        @brief Validate the center input.

        @param center: center of the ellipse.
        @return: validated center.
        """
        if center is None:
            return np.zeros(2)
        
        center = np.asarray(center)
        if center.shape != (2,):
            raise ValueError("Center shape should be (2,) as center is a 2D point.")
        return center
    
    def _construct_mesh(self) -> np.ndarray:
        """
        Construct the discretized mesh for the elliptical boundary.
        
        @return: Discretized boundary points and element indices.
        """
        # Generate angular points for discretization (0 to 2*pi)
        angles = np.linspace(0, 2 * np.pi, self.N_e + 1)
        
        # Parametric equation for an ellipse in 2D: 
        x = self.a * np.cos(angles) + self.center[0]
        y = self.b * np.sin(angles) + self.center[1]
        
        # Stack x and y to get the coordinates of boundary nodes
        nodes = np.vstack([x, y]).T
        
        # Create element indices
        elements = np.column_stack((np.arange(self.N_e), np.arange(1, self.N_e + 1)))
        
        return nodes, elements
    
    def contains(self, points: np.ndarray) -> np.ndarray:
        """
        @brief Check if the given points are inside the ellipse.
        
        @param points: Array of 2D points to check. Shape (N, 2) where N is the number of points.
        @return: Boolean array where True means the point is inside the ellipse, False otherwise.
        """
        points = np.asarray(points)
        if points.shape[1] != 2:
            raise ValueError("Points should have shape (N, 2).")

        # Use the ellipse equation (x/a)^2 + (y/b)^2 <= 1 to check if points are inside
        normalized_x = (points[:, 0] - self.center[0]) / self.a
        normalized_y = (points[:, 1] - self.center[1]) / self.b

        return (normalized_x ** 2 + normalized_y ** 2) <= 1
    
    def display(self, ax: plt.axes = None) -> None:
        ellipse_patch = mpatches.Ellipse(
            xy=self.center, width=2*self.a, height=2*self.b,
            facecolor=MSColors.GREY, edgecolor=MSColors.GREY, alpha=0.5, linestyle='--'
        )
        super().display(patch=ellipse_patch, ax=ax)

# Example usage
if __name__ == '__main__':
    N_e = 20
    a = 4.0  # semi-major axis
    b = 1.0  # semi-minor axis
    ellipse = Ellipse(N_e=N_e, a=a, b=b)
    ellipse.display()