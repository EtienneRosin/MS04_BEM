from helmoltz_2d_BEM.geometry import Disc
from helmoltz_2d_BEM.utils.quadratures.gauss_legendre_2_points import quadrature_points, quadrature_weights
from helmoltz_2d_BEM.utils.graphics import MSColors

import numpy as np
import matplotlib.pyplot as plt

import scienceplots
plt.style.use('science')
    
if __name__ == '__main__':
    N_e = 100
    a = 1
    
    lst_N = np.arange(start = 5, stop = N_e, step = 5)
    circle_perimeters = []
    
    for N in lst_N:
        disc = Disc(N_e = N, radius = a)
    
        y_e_m = disc.y_e_m
        y_e_d = disc.y_e_d
        
        value = 0
        for omega_q, x_q in zip(quadrature_weights, quadrature_points):
            value += np.sum(omega_q * np.linalg.norm(y_e_d, axis = 1))
        
        circle_perimeters.append(value)
    
    circle_perimeters = np.asarray(circle_perimeters)
    errors = np.abs(2*np.pi - circle_perimeters)/(2*np.pi)
    
    
    line_props = dict(marker = "o", markersize = 3, linestyle = "--", linewidth = 0.75)
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(lst_N, errors, **line_props)
    ax.set(xlabel = "$N_e$", ylabel = r"$\frac{\left|L - L_{N_e}\right|}{L}$")
    
    
    print(lst_N)
    print(errors)
    fig.savefig(fname="Figures/quadrature_validation.pdf")
    plt.show()
    
    # print(f"{value = }, 2 pi = {2*np.pi}")