from helmoltz_2d_BEM.integral_equation import IntegralEquation
from helmoltz_2d_BEM.integral_representation import IntegralRepresentation
from helmoltz_2d_BEM.utils import PlaneWave
from helmoltz_2d_BEM.geometry.domains import RectangularDomainWithObstacle
from helmoltz_2d_BEM.geometry.obstacles import Disc, Square

import numpy as np

boundaries = [[-4, 8], [-3, 3]]
steps = 10 * np.array([12, 6])
a = 1

N_e = 500

disc = Disc(N_e=N_e, radius=a)
square = Square(N_e=N_e, width=a)
# Plane wave --------------------------------
wave_number = 4*np.pi
k = wave_number * np.array([1, 1])

N = 1000
u_inc = PlaneWave(k=k)
IE = IntegralEquation(u_inc = PlaneWave(k = k), obstacle = square)
p = IE.solve()


Omega = RectangularDomainWithObstacle(boundaries=boundaries, steps=steps, obstacle=square)

IR = IntegralRepresentation(domain=Omega, u_inc=u_inc, normal_trace=p)

IR.display(field="real", save_name="Figures/square_case")
IR.display(field="angle", save_name="Figures/square_case")
IR.display(field="abs", save_name="Figures/square_case")