
from helmoltz_2d_BEM.geometry.obstacles import Obstacle
from helmoltz_2d_BEM.utils import MSColors

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Disc(Obstacle):
    def __init__(self, N_e: int, radius: float, center: list|np.ndarray = None) -> None:
        """
        @brief Constructor.
        
        @param N_e: number of Gamma_e of the boundary discretization.
        @param radius: radius of the circle.
        @param center: center of the circle.
        """
        # super().__init__(N_e)
        self.N_e = self._validate_N_e(N_e)
        self.radius = self._validate_radius(radius)
        self.center = self._validate_center(center)
        self.y_e, self.polar_nodes, self.Gamma_e = self._construct_mesh()
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
        if N_e <= 0:
            raise ValueError("N_e should be > 0.")
        return N_e
        
    def _validate_radius(self, radius: float) -> float:
        """
        @brief Validate the radius input.

        @param radius: radius of the disc.
        @return: validated radius.
        """
        if radius <= 0:
            raise ValueError("Radius should be > 0.")
        return radius
    
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
        lst_theta = np.linspace(start=-np.pi,stop=np.pi, num=self.N_e + 1, endpoint=True)
        R, THETA = np.meshgrid(self.radius, lst_theta, indexing='ij')
        polar_nodes = np.column_stack((R.ravel(), THETA.ravel()))
        XX, YY = self.radius * np.cos(THETA), self.radius * np.sin(THETA)
        y_e = np.column_stack((XX.ravel(), YY.ravel()))
        
        Gamma_e = np.column_stack((np.arange(self.N_e), np.arange(1, self.N_e + 1)))
        return y_e, polar_nodes, Gamma_e.astype(int)
    
    def contains(self, point: np.ndarray) -> np.ndarray:
        """
        @brief Check if the given point is inside the disc.

        @param point: 2D point to check.
        @return: True if the point is inside the disc, False otherwise.
        """
        return np.linalg.norm(point - self.center, axis=1) <= self.radius
 
    def display(self, ax: plt.axes = None) -> None:
        disc_patch = mpatches.Circle(
            xy=self.center,
            radius=self.radius,
            facecolor = MSColors.GREY,
            edgecolor = MSColors.GREY,
            alpha=0.5,
            linestyle='--'
        )
        super().display(patch=disc_patch, ax=ax)
        
        
if __name__ == '__main__':
    a = 1
    N_e = 6
    
    
    
    disc = Disc(N_e=N_e, radius=a)
    disc.display()
    
    print(disc.Gamma_e, "\n")
    print(disc.polar_nodes, "\n")
    
    print(disc.polar_nodes[disc.Gamma_e], "\n")
    # print(disc.a_e, "\n")
    # print(disc.b_e)
    