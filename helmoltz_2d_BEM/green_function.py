import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.special import jv, hankel1


def G(x: float|np.ndarray, y: float|np.ndarray, k: float) -> float|np.ndarray:
    """Green functiin of the 2D Helmoltz problem.

    Parameters:
        x (float | np.ndarray): x-direction position(s)
        y (float | np.ndarray): y-direction position(s)
        k (float): wave number of the incident wave

    Returns:
        (float|np.ndarray): value(s) of G(x,y)
    """
    return (1j/4)*hankel1(0, k * np.linalg.norm(x - y))

def G_polar(r: float|np.ndarray, k: float) -> float|np.ndarray:
    """Green functiin of the 2D Helmoltz problem.

    Parameters:
        r
        k (float): wave number of the incident wave

    Returns:
        (float|np.ndarray): value(s) of G(x,y)
    """
    return (1j/4)*hankel1(0, k * r)