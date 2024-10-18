from helmoltz_2d_BEM.utils import PlaneWave
from helmoltz_2d_BEM.circle_obstacle_case import ExactSolution
from helmoltz_2d_BEM.geometry import Disc
from helmoltz_2d_BEM.utils.graphics import MSColors

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

class ExactNormaTrace:
    """Represents the exact normal trace of the 2D Helmoltz problem.
    """
    def __init__(self, disc: Disc, u_inc: PlaneWave, u_plus: ExactSolution) -> None:
        self.disc = disc
        self.u_inc = u_inc
        self.u_plus = u_plus
        pass
    
    def __call__(self, x: float|np.ndarray) -> float|np.ndarray:
        """
        @brief Evaluate the normal trace normal

        @param x: points where the wave is evaluated. Should be of shape (2, N), where N is the number of points.
        @return: values of the normal trace derivative at x.
        """
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[0] != 2:
            raise ValueError("Input points should be of shape (2, N) for a 2D wave.")
        return - self.u_inc.radial_derivative(x) - self.u_plus.radial_derivative(x)
    
    def display(self, save_name: str = None) -> None:
        """
        @brief Display the normal trace on the disc boundary
        @param save_name: (Optional) save name of the figure if provided
        """
        polar_nodes = self.disc.polar_nodes
        
        lst_theta = polar_nodes.T[1].T[self.disc.Gamma_e].mean(axis = 1)
        
        values = self(self.disc.y_e_m.T)
        with plt.style.context('science' if save_name else 'default'):
            fig = plt.figure()
            ax = fig.add_subplot()
            
            ax.plot(lst_theta, values.real, label = r"$\Re (p)$")
            ax.plot(lst_theta, values.imag, label = r"$\Im (p)$")
            ax.set(xlabel = r"$\theta$")
            ax.legend()
            if save_name:
                fig.savefig(f"{save_name}.pdf")
            plt.show()

if __name__ == "__main__":
    N_e = 200
    a = 1
    
    # Plane wave --------------------------------
    wave_number = 2*np.pi
    k = wave_number * np.array([1, 0])
    
    # Exact solution ----------------------------
    N = 100
    
    # -------------------------------------------
    disc = Disc(N_e=N_e, radius=a)
    u_inc = PlaneWave(k = k)
    u_plus = ExactSolution(k = k, a = a, N = N)
    
    # -------------------------------------------
    p = ExactNormaTrace(disc=disc, u_inc=u_inc, u_plus=u_plus)
    
    p.display(save_name="Figures/exact_normal_trace")