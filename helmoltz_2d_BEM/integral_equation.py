from helmoltz_2d_BEM.geometry import Disc, Obstacle
from helmoltz_2d_BEM.utils import PlaneWave
from helmoltz_2d_BEM.integral_representation import ExactSolution
from helmoltz_2d_BEM.utils.quadratures.gauss_legendre_2_points import quadrature_points, quadrature_weights
from helmoltz_2d_BEM.utils.graphics import MSColors
from helmoltz_2d_BEM.utils.graphics.display_2d_complex_field import display_2d_complex_field
import helmoltz_2d_BEM.green_function as gf

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.special import jv, hankel1
import time


class IntegralEquation:
    r"""
    Class to represents the integral equation of the 2D Helmoltz problem with Dirichlet condition.

    Attributes
    ----------
    u_inc : PlaneWave
        incident plane wave
    obstacle : Obstacle
            obstacle
    p: np.ndarray
        normal trace solution of the integral equation
    A_assembly_time : float
        time to assembly the A matrix
    b_assembly_time : float
        time to assembly the b matrix
    solve_time : float
        time to solve compute the integrale representation
        
    Notes
    -----
    The integral equation is the following:

    .. math:: Find p \in H^{-1/2}(\Gamma) s.t. : \int_\Gamma G(\vec{x}, \vec{y})p(\vec{y}) d \Gamma(y) = - u^{\text{inc}}
    
    Whose variationnal formulation can be expressed as matricial system :
    .. math:: A\vec{p} = \vec{b}
    where : 
    - .. math:: p_i = p(\vec{y}_i)
    - .. math:: b_i = - \int_{\Gamma_i} u^{\text{inc}}(\vec{y}) d \Gamma(\vec{y})
    - .. math:: A_{ij} = \int_{\Gamma_j}\int_{\Gamma_i}G(\vec{x}, \vec{y})d \Gamma(\vec{x})d \Gamma(\vec{y})
    
    """
    
    def __init__(self, u_inc: PlaneWave, obstacle: Obstacle) -> None:
        r""" 
        Constructs the IntegralEquation object.

        Parameters
        ----------
        u_inc : PlaneWave
            incident plane wave
        obstacle : Obstacle
            obstacle
        """
        self.u_inc = u_inc
        self.obstacle = obstacle
        self.p = None
        self.A_assembly_time = 0.0
        self.b_assembly_time = 0.0
        self.solve_time = 0.0
        
    def _construct_b(self) -> np.ndarray:
        r""" 
        Constructs the right-hand side member of the variationnal formulation

        Returns
        -------
        : np.ndarray
            the vector b
        """
        start_time = time.time()
        y_e_m = self.obstacle.y_e_m 
        y_e_d = self.obstacle.y_e_d
        
        b = np.zeros((y_e_d.shape[0]), dtype=complex)        
        for omega_q, x_q in zip(quadrature_weights, quadrature_points):
            b += omega_q * (self.u_inc((x_q * y_e_d + y_e_m).T))
        b *= - np.linalg.norm(y_e_d, axis = 1)
        self.b_assembly_time = time.time() - start_time
        return b
        # return - np.linalg.norm(y_e_d, axis = 1) * b 
    
    def _construct_A(self) -> np.ndarray:
        r""" 
        Constructs the matrix of the variationnal formulation

        Returns
        -------
        : np.ndarray
            the matrix A
        """
        start_time = time.time()
        y_e_m = self.obstacle.y_e_m
        a_e = self.obstacle.a_e
        b_e = self.obstacle.b_e
        y_e_d = self.obstacle.y_e_d
        k = self.u_inc.wave_number

        A = np.zeros((a_e.shape[0], a_e.shape[0]), dtype = complex)
        RP = 1j / 4 - (np.log(k / 2) + np.euler_gamma) / (2 * np.pi) 
        for i in range(a_e.shape[0]):
            for j in range(a_e.shape[0]):
                if i != j:
                    for omega_q, x_q in zip(quadrature_weights, quadrature_points):
                        for omega_q_p, x_q_p in zip(quadrature_weights, quadrature_points):
                            point_i = x_q * y_e_d[i] + y_e_m[i]
                            point_j = x_q_p * y_e_d[j] + y_e_m[j]
                            A[i, j] += omega_q * omega_q_p * gf.G(point_i, point_j, k)
                            
                    A[i, j] = np.linalg.norm(y_e_d[i]) * np.linalg.norm(y_e_d[j]) * A[i, j]
                    
                else:
                    gamma_e = 2 * np.linalg.norm(y_e_d[i])
                    for omega_q, x_q in zip(quadrature_weights, quadrature_points):
                        B_e = b_e[i] - (x_q * y_e_d[i] + y_e_m[i])
                        A_e = a_e[i] - (x_q * y_e_d[i] + y_e_m[i])
                        
                        A[i, i] += - (omega_q / (2 * np.pi)) * (np.dot(B_e, y_e_d[i]) * np.log(np.linalg.norm(B_e)) - np.dot(A_e, y_e_d[i]) * np.log(np.linalg.norm(A_e)))
                    A[i, i] =  A[i,j] + (1/(2 * np.pi) + RP) * (gamma_e ** 2)
        self.A_assembly_time = time.time() - start_time
        return A

    
    def solve(self) -> np.ndarray:
        r""" 
        Solve the variationnal formulation.

        Returns
        -------
        p : np.ndarray
            the vector p solution
        """
        
        A = self._construct_A()
        b = self._construct_b()
        start_time = time.time()
        p = sp.linalg.solve(A, b)
        self.p = p
        self.solve_time = time.time() - start_time
        return p
    
    # def display(self) -> None:
    #     U = self.solve()
    #     polar_nodes = self.obstacle.polar_nodes
        
    #     lst_theta = polar_nodes.T[1].T[self.obstacle.Gamma_e].mean(axis = 1)
    #     fig = plt.figure()
    #     ax = fig.add_subplot()
    #     ax.plot(lst_theta, U.real, label = "real")
    #     ax.plot(lst_theta, U.imag, label = "imag")
    #     ax.legend() 
    #     plt.show()
        
    def display(self, field = "real", save_name: str = None) -> None:
        """
        @brief Display the normal trace on the disc boundary
        @param save_name: (Optional) save name of the figure if provided
        """
        
        
        if self.p is None:
            self.solve()
        
        
        
        values = self.p
        display_2d_complex_field(*self.obstacle.y_e_m.T, values, field=field, save_name=save_name)
if __name__ == '__main__':
    N_e = 200
    a = 1
    wave_number = 2*np.pi
    k = wave_number * np.array([1, 0])
    disc = Disc(N_e=N_e, radius=a)
    
    i_e = IntegralEquation(u_inc = PlaneWave(k = k), obstacle = disc)
    i_e.display(field = "angle")