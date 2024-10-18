

import numpy as np
import matplotlib.pyplot as plt
# from helmoltz_2d_BEM.utils.graphics import MSColors
# import matplotlib.patches as mpatches


from helmoltz_2d_BEM.geometry.domains import RectangularDomain
from helmoltz_2d_BEM.geometry.obstacles import Obstacle, Disc, Square

class RectangularDomainWithObstacle(RectangularDomain):
    def __init__(self, boundaries: list|np.ndarray, steps: int|list|np.ndarray, obstacle: Obstacle) -> None:
        super().__init__(boundaries, steps)
        self.obstacle = obstacle
        self._mask_node()
        
    def _mask_node(self) -> None:
        """
        @brief Filter out nodes that are inside the obstacles.
        """
        mask = np.zeros(self.nodes.shape[0], dtype=bool)
        mask |= self.obstacle.contains(self.nodes)
        if np.all(mask): 
            raise ValueError("All nodes are inside the obstacle! Adjust the geometry.")
        self.nodes = self.nodes[~mask]
        
    def display(self, ax: plt.axes = None):
        fig = plt.figure()
        ax = fig.add_subplot()
        super().display(ax)
        self.obstacle.display(ax=ax)
        ax.set(xlabel = r"$x$", ylabel = r"$y$", aspect = "equal")
        plt.show()
        
if __name__ == '__main__':
    boundaries = [[-2, 5], [-1.5, 1.5]]
    steps = [20, 20]
    N_e = 10
    a = 1
    disc = Disc(N_e= N_e, radius=a)
    square = Square(N_e=N_e, width=a)
    
    Omega = RectangularDomainWithObstacle(boundaries=boundaries, steps=steps, obstacle=disc)
    Omega.display()