
import numpy as np
import matplotlib.pyplot as plt
from helmoltz_2d_BEM.utils import PlaneWave
from helmoltz_2d_BEM.utils.graphics.display_2d_complex_field import display_2d_complex_field
from helmoltz_2d_BEM.integral_equation import IntegralEquation
from helmoltz_2d_BEM.geometry.obstacles import Disc
from helmoltz_2d_BEM.circle_obstacle_case import ExactSolution, ExactNormaTrace



# Domain ------------------------------------
boundaries = [[-4, 8], [-3, 3]]
steps = 10 * np.array([12, 6])

# Disc --------------------------------------
a = 1
lst_N_e = np.arange(start=100, step=100, stop=2500)

N_e = 500

if __name__ == '__main__':
    disc = Disc(N_e=N_e, radius=a)
    # Plane wave --------------------------------
    wave_number = 2*np.pi
    k = wave_number * np.array([1, 0])

    # Exact solution ----------------------------
    N = 100
    u_inc = PlaneWave(k=k)
    u_plus = ExactSolution(k=k, a=a, N=N)
    p = ExactNormaTrace(disc=disc, u_inc=u_inc, u_plus=u_plus)
    

    # Integral representation -------------------
    i_e = IntegralEquation(u_inc = PlaneWave(k = k), obstacle = disc)
    p = i_e.solve()
    # i_e.display(save_name="Figures/IE_trace")
    
    polar_nodes = disc.polar_nodes
    lst_theta = polar_nodes.T[1].T[disc.Gamma_e].mean(axis = 1)

    save_name = "Figures/IE_trace"
    
    
    with plt.style.context('science' if save_name else 'default'):
        fig = plt.figure()
        ax = fig.add_subplot()
        
        ax.plot(lst_theta, p.real, label = r"$\Re (p)$")
        ax.plot(lst_theta, p.imag, label = r"$\Im (p)$")
        ax.set(xlabel = r"$\theta$")
        ax.legend()
        if save_name:
            fig.savefig(f"{save_name}.pdf")
        plt.show()