
from helmoltz_2d_BEM.geometry.obstacles import Obstacle
from helmoltz_2d_BEM.utils import MSColors

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches




class Square(Obstacle):
    """
    Class to represents a square obstacle.

    Attributes
    ----------
    N_e : int
        number of discretized elements
    width : float
        width of the square
    center : list|np.ndarray
        center of the square
    y_e: np.ndarray
        nodes of the discretization
    Gamma_e: np.ndarray
        elements indices
    a_e: np.ndarray
        elements's firt node
    b_e: np.ndarray
        elements's last node
    y_e_m: np.ndarray
        elements's middle point
    y_e_d: np.ndarray
        elements's difference point
    """
    def __init__(self, N_e: int, width: float, center: list|np.ndarray = None) -> None:
        """
        Constructs the square object.

        Parameters
        ----------
            N_e : int
                number of elements of the boundary discretization
            width : float
                width of the square
            center : list|np.ndarray
                center of the square
        """
        self.N_e = self._validate_N_e(N_e)
        self.width = self._validate_width(width)
        self.center = self._validate_center(center)
        
        self.y_e, self.Gamma_e = self._construct_mesh()
        self.a_e = self.y_e[:-1]
        self.b_e = self.y_e[1:]
        
        self.y_e_m = np.mean(self.y_e[self.Gamma_e], axis=1)
        self.y_e_d = 0.5 * (self.b_e - self.a_e)
        
    def _validate_N_e(self, N_e: int) -> float:
        """
        @brief Validate the N_e input.

        @param N_e: number of elements for the discretization.
        @return: validated number of elements.
        """
        if N_e <= 4:
            raise ValueError("N_e should be >= 4.")
        return N_e
        
    def _validate_width(self, width: float) -> float:
        """
        @brief Validate the width input.

        @param width: width of the square.
        @return: validated width.
        """
        if width <= 0:
            raise ValueError("width should be > 0.")
        return width
    
    def _validate_center(self, center: list|np.ndarray) -> float:
        """
        @brief Validate the center input.

        @param center: center of the disc.
        @return: validated center.
        """
        if center is None:
            return np.zeros(2)
        
        center = np.asarray(center)
        if center.shape != (2,):
            raise ValueError("Center shape should be (2,) as center is a 2D point.")
        return center
    
    def _construct_mesh(self) -> np.ndarray:
        half_width = self.width / 2
        corners = [
            [self.center[0] - half_width, self.center[1] - half_width],  # bas gauche
            [self.center[0] + half_width, self.center[1] - half_width],  # bas droit
            [self.center[0] + half_width, self.center[1] + half_width],  # haut droit
            [self.center[0] - half_width, self.center[1] + half_width],  # haut gauche
        ]
        
        segments_per_side = [self.N_e // 4] * 4
        remaining_segments = self.N_e % 4
        
        for i in range(remaining_segments):
            segments_per_side[i] += 1
        
        nodes = []
        for i in range(4):
            start_corner = np.array(corners[i])
            end_corner = np.array(corners[(i + 1) % 4])
            side_points = np.linspace(start=start_corner, stop=end_corner, num=segments_per_side[i], endpoint=False)
            nodes.append(side_points)
        
        nodes = np.vstack(nodes)
        nodes = np.vstack([nodes, nodes[0]])  # Fermer le contour pour avoir N_e + 1 points
        
        elements = np.column_stack((np.arange(self.N_e), np.arange(1, self.N_e + 1)))
        
        return nodes, elements
    
    def contains(self, points: np.ndarray) -> np.ndarray:
        """
        @brief Check if the given points are inside the square.
        
        @param points: Array of 2D points to check. Shape (N, 2) where N is the number of points.
        @return: Boolean array where True means the point is inside the square, False otherwise.
        """
        # Vérifie que `points` est bien un tableau de forme (N, 2)
        points = np.asarray(points)
        if points.shape[1] != 2:
            raise ValueError("Points should have shape (N, 2).")

        # Vérifie si chaque point est dans les limites du carré en utilisant des opérations vectorisées
        inside_x = (self.center[0] - self.width / 2 <= points[:, 0]) & (points[:, 0] <= self.center[0] + self.width / 2)
        inside_y = (self.center[1] - self.width / 2 <= points[:, 1]) & (points[:, 1] <= self.center[1] + self.width / 2)

        return inside_x & inside_y
    
    def display(self, ax: plt.axes = None) -> None:
        square_patch = mpatches.Rectangle(
            xy=[self.center[0] - self.width / 2, self.center[1] - self.width / 2],
            width=self.width, height=self.width,
            facecolor=MSColors.GREY, edgecolor=MSColors.GREY, alpha=0.5, linestyle='--'
        )
        super().display(patch=square_patch, ax=ax)
        
if __name__ == '__main__':
    N_e = 8
    width = 1
    square = Square(N_e=N_e, width=width)
    square.display()