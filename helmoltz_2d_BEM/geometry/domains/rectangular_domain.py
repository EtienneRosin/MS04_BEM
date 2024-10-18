import numpy as np
import matplotlib.pyplot as plt
from helmoltz_2d_BEM.utils.graphics import MSColors
import matplotlib.patches as mpatches

class RectangularDomain:
    """
    Class to represents a rectangular domain.

    Attributes
    ----------
    boundaries: np.ndarray
        boundaries of the domain
    steps: np.ndarray
        step size in each direction
    nodes: np.ndarray
        mesh nodes
    """
    def __init__(self, boundaries: list|np.ndarray, steps: int|list|np.ndarray) -> None:
        """
        Constructs the rectangular domain object.

        Parameters
        ----------
        boundaries: np.ndarray
            boundaries of the domain
        steps: np.ndarray
            step size in each direction
        """
        self.boundaries = self._validate_boundaries(boundaries)
        self.steps = self._validate_steps(steps)
        self.nodes = self._construct_mesh()
    
    def _validate_boundaries(self, boundaries: list|np.ndarray) -> np.ndarray:
        """
        @brief Validate the boundaries input.

        @param boundaries: ArrayLike, boundaries of the rectangular domain.
        @return: np.ndarray, validated boundaries.
        """
        boundaries = np.asarray(boundaries)
        if boundaries.shape != (2, 2):
            raise ValueError(f"Boundaries shape (here: {boundaries.shape}) should be (2, 2) for a 2D domain.")
        return boundaries
    
    def _validate_steps(self, steps: int|list|np.ndarray) -> np.ndarray:
        """
        @brief Validate the steps input.

        @param steps: steps for the mesh (if an integer is given, the mesh would be regular).
        @return: validated steps.
        """
        steps = np.asarray(steps)
        
        if isinstance(steps, int):
            if steps <= 0:
                raise ValueError("Steps should be > 0.")
            return np.array([steps, steps])  # Convertir un entier en un tableau 2D régulier.
        steps = np.asarray(steps)
        
        if steps.shape != (2,):
            raise ValueError(f"Steps shape (here: {steps.shape}) should be (2,) for a 2D domain.")
        return steps
    
    def _construct_mesh(self) -> np.ndarray:
        """
        @brief Construct the mesh of the rectangular domain.

        @return: the mesh.
        """
        
        lst_x = np.linspace(*self.boundaries[0,:], num=self.steps[0], endpoint=True)
        lst_y = np.linspace(*self.boundaries[1,:], num=self.steps[1], endpoint=True)
        xx, yy = np.meshgrid(lst_x, lst_y, indexing='ij')
        return np.column_stack((xx.ravel(), yy.ravel()))
    
    def display(self, ax: plt.axes = None) -> None:
        """
        Display the rectangular domain.

        Parameters
        ----------
        ax: plt.axes, default = None 
            axes onto which to display the obstacle
        
        Raises
        ------
        """
        show = False
        if ax is None:
            fig, ax = plt.subplots()
            ax.set(xlabel = r"$x$", ylabel = r"$y$", aspect = "equal")
            show = True
            
        rect = mpatches.Rectangle(
            (self.boundaries[0, 0], self.boundaries[1, 0]),
            self.boundaries[0, 1] - self.boundaries[0, 0],
            self.boundaries[1, 1] - self.boundaries[1, 0],
            facecolor=MSColors.LIGHT_BLUE,
            edgecolor=MSColors.LIGHT_BLUE,
            alpha=0.5
        )
        ax.add_patch(rect)
        
        ax.scatter(*self.nodes.T, s = 5,c = MSColors.LIGHT_BLUE)
        if show:
            plt.show()
            
if __name__ == '__main__':
    boundaries = [[-2, 5], [-1.5, 1.5]]
    steps = [20, 20]
    
    Omega = RectangularDomain(boundaries=boundaries, steps=steps)
    Omega.display()