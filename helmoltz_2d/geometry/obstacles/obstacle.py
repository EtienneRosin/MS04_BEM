from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class Obstacle(ABC):
    """
    Abstract class to handle different obstacle geometries.
    """
    def __init__(self, N_e: int) -> None:
        """
        @brief Constructor for the Obstacle class.

        @param N_e: number of elements for the discretization.
        """
        self.N_e = self._validate_N_e(N_e)
        # super().__init__()
        
    def _validate_N_e(self, N_e: int) -> float:
        """
        @brief Validate the radius input.

        @param N_e: number of elements for the discretization.
        @return: validated number of elements.
        """
        if N_e <= 0:
            raise ValueError("N_e should be > 0.")
        return N_e
        
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
        
    @abstractmethod
    def display(self, ax: plt.axes = None) -> None:
        """
        @brief Display the obstacle.

        @param ax: axes onto which to display the obstacle (default: None).
        """
        pass