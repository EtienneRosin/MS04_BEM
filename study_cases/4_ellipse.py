from helmoltz_2d_BEM.integral_equation import IntegralEquation
from helmoltz_2d_BEM.integral_representation import IntegralRepresentation
from helmoltz_2d_BEM.utils import PlaneWave
from helmoltz_2d_BEM.geometry.domains import RectangularDomainWithObstacle
from helmoltz_2d_BEM.geometry.obstacles import Disc, Square, Ellipse

import numpy as np

boundaries = [[-4, 8], [-3, 3]]
steps = 10 * np.array([12, 6])
a = 1

N_e = 500

a = 2.0  # semi-major axis
b = 0.5  # semi-minor axis
ellipse = Ellipse(N_e=N_e, a=a, b=b)
# Plane wave --------------------------------
wave_number = 4*np.pi
direction = np.array([2, 1])
k = wave_number * direction / np.linalg.norm(direction)

N = 1000
u_inc = PlaneWave(k=k)
IE = IntegralEquation(u_inc = PlaneWave(k = k), obstacle = ellipse)
p = IE.solve()


Omega = RectangularDomainWithObstacle(boundaries=boundaries, steps=steps, obstacle=ellipse)

IR = IntegralRepresentation(domain=Omega, u_inc=u_inc, normal_trace=p)

IR.display(field="real", 
        #    save_name="Figures/ellipse_case"
           )
IR.display(field="angle",
        #    save_name="Figures/ellipse_case"
           )
IR.display(field="abs", 
        #    save_name="Figures/ellipse_case"
           )