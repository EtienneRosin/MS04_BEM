
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, hankel1

from helmoltz_2d_BEM.utils.graphics import MSColors
from helmoltz_2d_BEM.geometry.domains import RectangularDomain
from helmoltz_2d_BEM.utils.graphics import prepare_points_for_pcolormesh



class ExactSolution:
    r"""
    @class ExactSolution
    @brief Represent the diffracted wave from a plane wave by a disc of radius a (it is the exact solution of the 2D Helmoltz equation with,  Dirichlet condition of a plane wave ^{-i \vec{k} \cdot \vec{x}}).
    """
    def __init__(self, k: list|np.ndarray, a: float, N: int) -> None:
        """
        @brief Constructor.
        
        @param k: wave vector of the initial plane wave.
        @param a: radius of the disc
        @param N: number of terms of the approximation.
        """
        self.k = self._validate_k(k)
        self.a = self._validate_a(a)
        self.N = self._validate_N(N)
        self.wave_number = np.linalg.norm(self.k)
        self.n_values = np.arange(-self.N, self.N + 1).reshape(-1, 1)  # Create a range of n values
        self.A_n = ((-1j)**self.n_values) * jv(self.n_values, self.wave_number * self.a) / hankel1(self.n_values, self.wave_number * self.a)
        
    def _validate_k(self, k: list|np.ndarray) -> np.ndarray:
        """
        @brief Validate the k input.

        @param k: wave vector of the initial plane wave.
        @return: wave vector.
        """
        k = np.asarray(k)
        if k.shape != (2,):
            raise ValueError("k shape should be (2,) as it is a 2d vector.")
        return k
    
    def _validate_a(self, a: float) -> float:
        """
        @brief Validate the radius input.

        @param a: radius of the disc.
        @return: validated radius.
        """
        if a <= 0:
            raise ValueError("Radius should be > 0.")
        return a
    
    def _validate_N(self, N: int) -> int:
        """
        @brief Validate the N input.

        @param N: number of terms of the approximation.
        @return: validated  number of terms.
        """
        if N <= 0:
            raise ValueError("N should be > 0.")
        return N
    
    
    def __call__(self, x: list|np.ndarray) -> np.ndarray:
        """
        @brief Evaluate the exact solution at the given points.

        @param x: points where the wave is evaluated. Should be of shape (2, N), where N is the number of points.
        @return: values of the exact solution at x.
        """
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[0] != 2:
            raise ValueError("Input points should be of shape (2, N) for a 2D wave.")
        
        # Compute the radial distance r
        r = np.linalg.norm(x, axis=0)
        cos_theta = np.dot(self.k, x) / (self.wave_number * r)
        theta = np.arccos(cos_theta)
        
        hankel_terms = hankel1(self.n_values, self.wave_number * r)
        exp_terms = np.exp(1j * self.n_values * theta) 

        return - np.sum(self.A_n * hankel_terms * exp_terms, axis=0)
    
    def radial_derivative(self, x: list|np.ndarray) -> np.ndarray:
        r"""
        @brief Evaluate \partial_r of the exact solution at x.

        @param x: points where \partial_r is evaluated. Should be of shape (2, N), where N is the number of points.
        @return: values of \partial_r of the exact solution at x.
        """ 
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[0] != 2:
            raise ValueError("Input points should be of shape (2, N) for a 2D wave.")
        
        # Compute the radial distance r
        r = np.linalg.norm(x, axis=0)
        cos_theta = np.dot(self.k, x) / (self.wave_number * r)
        # print(r)
        theta = np.arccos(cos_theta)
        
        hankel_derivative_terms = hankel1(self.n_values - 1, self.wave_number * r) - hankel1(self.n_values + 1, self.wave_number * r)
        
        exp_terms = np.exp(1j * self.n_values * theta)
        
        return - 0.5 * self.wave_number * np.sum(self.A_n * hankel_derivative_terms * exp_terms, axis=0)
    
    def display(self, ax: plt.axes = None, boundaries=[[-10, 10], [-10, 10]], steps=[100, 100], field = "real"):
        """
        @brief Display the exact solution over a rectangular domain.

        @param ax: Optional, matplotlib axes object.
        @param boundaries: Optional, list specifying the x and y boundaries of the domain.
        @param steps: Optional, list specifying the number of steps for the mesh grid in x and y directions.
        @param part: Optional, choose 'real' or 'imag' or 'abs' or 'angle' to display the respective property of the wave.
        """
        show = False
        if ax is None:
            fig, ax = plt.subplots()
            ax.set(xlabel = r"$x$", ylabel = r"$y$", aspect = 'equal')
            show = True 
            
        domain = RectangularDomain(boundaries=boundaries, steps=steps)
        
        values = self(domain.nodes.T)
        X, Y, Z = prepare_points_for_pcolormesh(*domain.nodes.T, values)
        
        match field:
            case "real":
                pcm = ax.pcolormesh(X, Y, np.real(Z))
            case "imag":
                pcm = ax.pcolormesh(X, Y, np.imag(Z))
            case "abs":
                pcm = ax.pcolormesh(X, Y, np.abs(Z))
            case "angle":
                pcm = ax.pcolormesh(X, Y, np.angle(Z))
            case _:
                raise ValueError("Invalid part argument. Choose 'real' or 'imag' or 'abs' or 'angle'.")
        
        fig.colorbar(pcm, ax = ax)
        if show:
            plt.show()

if __name__ == '__main__':
    wave_number = 2
    k = wave_number * np.array([1, 1/2])
    a = 3
    N = 2
    
    u_plus = ExactSolution(k = k, a = a, N = N)
    u_plus.display(field='abs')
    
    # print(u_plus(np.array([[1, 1], [1, 2]])).shape)
    # print(f"{k.shape = }")
    # steps = [100, 100]
    # N = 100
    # field = 'real'
    # u_inc = PlaneWave(k = k)
    # u_inc.display(steps=steps, field=field)
    # u_inc.display_approx(N = N, steps=steps, field=field)
    # u_inc.display_comparison_with_approx(N = N, steps=steps, field=field)