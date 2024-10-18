from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


from helmoltz_2d_BEM.utils import MSColors

import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

class Obstacle(ABC):
    """
    Abstract class to represents different obstacle geometries.

    Attributes
    ----------
    N_e : int
        number of discretized elements
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
    N_e: str                # number of discretized elements
    y_e: np.ndarray         # nodes of the discretization
    Gamma_e: np.ndarray     # elements indices
    a_e: np.ndarray         # elements's firt node
    b_e: np.ndarray         # elements's last node
    y_e_m: np.ndarray       # elements's middle point
    y_e_d: np.ndarray       # elements's difference point
        
    @abstractmethod
    def _construct_mesh(self) -> tuple:
        """
        @brief Construct the mesh of the rectangular domain.

        @return: nodes, polar_nodes, elements.
        """
    
    @abstractmethod
    def contains(self, point: np.ndarray) -> np.ndarray:
        """
        @brief Determine if the obstacle contains a given point.

        @param point: np.ndarray, given point or points.
        @return: np.ndarray, boolean array indicating if the obstacle contains the given points.
        """
        pass
        
    # @abstractmethod
    # def display(self, ax: plt.axes = None) -> None:
    #     """
    #     @brief Display the obstacle.

    #     @param ax: axes onto which to display the obstacle (default: None).
    #     """
    #     pass
    def display(self, patch: mpatches.Patch = None, ax: plt.axes = None) -> None:
        show = False
        if ax is None:
            fig, ax = plt.subplots()
            ax.set(xlabel = r"$x$", ylabel = r"$y$", aspect = 'equal')
            show = True

        if patch:
            ax.add_patch(patch)
        
        ax.scatter(*self.y_e.T, label='Nodes', c = MSColors.DARK_BLUE, s = 10)        
        segments = self.y_e[self.Gamma_e]
        lc = LineCollection(segments, color = MSColors.DARK_BLUE, label = "Elements")
        ax.add_collection(lc)

        if show:
            ax.scatter(*self.y_e_m.T, label = "Elements middle", c = MSColors.RED, s = 10, zorder = 2)
            ax.scatter(*(self.y_e_d).T, label = "Elements diff", c = MSColors.ORANGE, s = 10, zorder = 2)
            ax.quiver(*(self.a_e).T, *(2*self.y_e_d).T, color = MSColors.GREEN, zorder = 2, angles = "xy", scale_units = 'xy',scale=1)
            
            ax.legend()
            plt.show()