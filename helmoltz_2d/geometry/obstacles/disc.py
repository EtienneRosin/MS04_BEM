
from helmoltz_2d.geometry.obstacles import Obstacle
# from helmoltz_2d.geometry.obstacles import Obstacle
from helmoltz_2d.utils import MSColors

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection




class Disc(Obstacle):
    def __init__(self, N_e: int, radius: float, center: list|np.ndarray = None) -> None:
        """
        @brief Constructor.
        
        @param N_e: number of elements of the boundary discretization.
        @param radius: radius of the circle.
        @param center: center of the circle.
        """
        super().__init__(N_e)
        self.radius = self._validate_radius(radius)
        self.center = self._validate_center(center)
        self.nodes, self.polar_nodes, self.elements = self._construct_mesh()
        self.elements_first_node = self.nodes[:-1]
        self.middle_nodes = np.mean(self.nodes[self.elements], axis=1)
        
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
        nodes = np.column_stack((XX.ravel(), YY.ravel()))
        
        elements = np.column_stack((np.arange(self.N_e), np.roll(np.arange(self.N_e), -1)))
        return nodes, polar_nodes, elements.astype(int)
    
    def contains(self, point: np.ndarray) -> np.ndarray:
        """
        @brief Check if the given point is inside the disc.

        @param point: 2D point to check.
        @return: True if the point is inside the disc, False otherwise.
        """
        return np.linalg.norm(point - self.center, axis=1) <= self.radius
        # return np.linalg.norm(point - self.center) <= self.radius
    
    def display(self, ax: plt.axes = None) -> None:
        show = False
        if ax is None:
            fig, ax = plt.subplots()
            ax.set(xlabel = r"$x$", ylabel = r"$y$", aspect = 'equal')
            show = True

        # Tracer le contour du disque
        disc_patch = mpatches.Circle(
            xy=self.center,
            radius=self.radius,
            facecolor = MSColors.GREY,
            edgecolor = MSColors.GREY,
            alpha=0.5,
            linestyle='--'
        )
        ax.add_patch(disc_patch)
        
        # Tracer les nœuds
        ax.scatter(self.nodes[:, 0], self.nodes[:, 1], label='Nodes', c = MSColors.DARK_BLUE, s = 10)        
        segments = self.nodes[self.elements]
        lc = LineCollection(segments, color=MSColors.DARK_BLUE, label = "Elements")
        ax.add_collection(lc)

        if show:
            ax.scatter(*self.middle_nodes.T, label = "Elements middle", c = MSColors.RED, s = 10, zorder = 2)
            ax.legend()
            plt.show()
        
if __name__ == '__main__':
    a = 1
    N_e = 10
    
    
    
    disc = Disc(N_e=N_e, radius=a)
    disc.display()