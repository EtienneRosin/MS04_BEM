""" Implementation of the IntegralRepresentation class.
"""
from helmoltz_2d_BEM.geometry.domains import RectangularDomainWithObstacle
from helmoltz_2d_BEM.utils import PlaneWave
import helmoltz_2d_BEM.green_function as gf
from helmoltz_2d_BEM.utils.quadratures.gauss_legendre_2_points import quadrature_points, quadrature_weights
from helmoltz_2d_BEM.utils.graphics.display_2d_complex_field import display_2d_complex_field

from helmoltz_2d_BEM.geometry.obstacles import Disc
from helmoltz_2d_BEM.circle_obstacle_case import ExactSolution, ExactNormaTrace

import numpy as np
import time


class IntegralRepresentation:
    r"""
    Class to represents the integral representation of the 2D Helmoltz problem with Dirichlet condition.

    Attributes
    ----------
    domain : RectangularDomainWithObstacle
        domain with an obstacle inside
    u_inc : PlaneWave
        incident plane wave
    normal_trace : np.ndarray
        array of the normal trace evaluated on the obstacle elements's middle (P_0 interpolation)
    G_assembly_time : float
        time to assembly the G matrix
    solve_time : float
        time to solve compute the integrale representation
        
    Notes
    -----
    The integral representation is given by:

    .. math:: u^+(\boldsymbol{x}) = \int_\Gamma G(\boldsymbol{x}, \boldsymbol{y})p(\boldsymbol{y}) d\Gamma(\boldsymbol{y})
    """
    def __init__(self, domain: RectangularDomainWithObstacle, u_inc: PlaneWave, normal_trace: np.ndarray) -> None:
        r""" 
        Constructs the IntegralRepresentation object.

        Parameters
        ----------
        domain : RectangularDomainWithObstacle
            domain with an obstacle inside
        u_inc : PlaneWave
            incident plane wave
        normal_trace : np.ndarray
            array of the normal trace evaluated on the obstacle elements's middle (P_0 interpolation)
        """
        self.domain = domain
        self.u_inc = u_inc
        self.p = self._validate_p(normal_trace)
        self.G_assembly_time = 0.0
        self.solve_time = 0.0
        self.U = None

    def _validate_p(self, normal_trace: np.ndarray) -> np.ndarray:
        """
        Validate the normal trace array

        Parameters
        ----------
        normal_trace : np.ndarray
            array of the normal trace evaluated on the obstacle elements's middle (P_0 interpolation)
        
        Returns
        -------
        normal_trace: np.ndarray
            array of the normal trace array if it is valide
        """
        if normal_trace.shape[0] != len(self.domain.obstacle.Gamma_e):
            raise ValueError("normal_trace should have the same shape as the obstacle elements")
        return normal_trace
    
    def _construct_G(self) -> np.ndarray:
        start_time = time.time()
        k = self.u_inc.wave_number
        x_i = self.domain.nodes
        y_e_m = self.domain.obstacle.y_e_m
        y_e_d = self.domain.obstacle.y_e_d

        G = np.zeros((x_i.shape[0], y_e_m.shape[0]), dtype=complex)

        for omega_q, x_q in zip(quadrature_weights, quadrature_points):
            # Quadrature sur les éléments avec une vérification de la dimension
            shifted_y = x_q * y_e_d + y_e_m  # Point quadrature sur l'élément
            r = np.linalg.norm(x_i[:, np.newaxis, :] - shifted_y[np.newaxis, :, :], axis=2)

            G += omega_q * np.linalg.norm(y_e_d, axis=1) * gf.G_polar(r, k)
        self.G_assembly_time = time.time() - start_time
        return G
    
    def solve(self) -> np.ndarray:
        start_time = time.time()
        self.U = self._construct_G() @ self.p
        self.solve_time = time.time() - start_time
        return self.U
    
    def display(self, field: str = "real", save_name: str = None) -> None:
        """
        @brief Display the approximate solution over the domain.
        
        @param field: (Optional) choose 'real' or 'imag' or 'abs' or 'angle' to display the respective property of the solution.
        @param save_name: (Optional) save name of the figure if provided 
        """
        if self.U is None:
            self.solve()
        display_2d_complex_field(*self.domain.nodes.T, self.U, field = field, save_name = save_name)
    
    
if __name__ == '__main__':
    
    # Domain ------------------------------------
    boundaries = [[-4, 8], [-3, 3]]
    steps = 5 * np.array([12, 6])
    
    # Disc --------------------------------------
    N_e = 60
    a = 1
    disc = Disc(N_e= N_e, radius=a)
    Omega = RectangularDomainWithObstacle(boundaries=boundaries, steps=steps, obstacle=disc)
    
    # Plane wave --------------------------------
    wave_number = 2*np.pi
    k = wave_number * np.array([1, 1])
    
    # Exact solution ----------------------------
    N = 100
    u_inc = PlaneWave(k = k)
    u_plus = ExactSolution(k = k, a = a, N=N)
    
    p = ExactNormaTrace(disc=disc, u_inc=u_inc, u_plus=u_plus)
    
    
    integral_representation = IntegralRepresentation(domain = Omega, u_inc=u_inc, normal_trace= p(disc.y_e_m.T))
    integral_representation.display()
    