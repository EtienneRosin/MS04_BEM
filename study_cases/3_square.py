from helmoltz_2d_BEM.integral_equation import IntegralEquation
from helmoltz_2d_BEM.integral_representation import IntegralRepresentation
from helmoltz_2d_BEM.utils import PlaneWave
from helmoltz_2d_BEM.geometry.domains import RectangularDomainWithObstacle
from helmoltz_2d_BEM.geometry.obstacles import Disc, Square

import numpy as np

import matplotlib.pyplot as plt
import scienceplots
import cmasher as cmr

from helmoltz_2d_BEM.utils import prepare_points_for_pcolormesh

boundaries = [[-4, 8], [-3, 3]]
steps = 20 * np.array([12, 6])
a = 1

N_e = 1000

disc = Disc(N_e=N_e, radius=a)
square = Square(N_e=N_e, width=a)
# Plane wave --------------------------------
wave_number = 4*np.pi
k = np.array([1, 0])
k = wave_number * k / np.linalg.norm(k)
N = 500
u_inc = PlaneWave(k=k)
IE = IntegralEquation(u_inc = PlaneWave(k = k), obstacle = square)
p = IE.solve()


Omega = RectangularDomainWithObstacle(boundaries=boundaries, steps=steps, obstacle=square)

IR = IntegralRepresentation(domain=Omega, u_inc=u_inc, normal_trace=p)
IR.solve()
IR.display(field="real", 
        #    save_name="Figures/square_case"
           )
IR.display(field="angle", 
        #    save_name="Figures/square_case"
           )
IR.display(field="abs", 
        #    save_name="Figures/square_case"
           )

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

X, Y, Z = prepare_points_for_pcolormesh(*IR.domain.nodes.T, IR.U)

XX, YY = np.meshgrid(X, Y)
ax.plot_surface(XX,YY, np.real(Z))
ax.set_aspect("equal")
plt.show()
# match field:
# case "real":
#     pcm = ax.pcolormesh(X, Y, np.real(Z), cmap = cmap)
#     label = r"$\Re\left(\tilde{u}^+\right)$"
# case "imag":
#     pcm = ax.pcolormesh(X, Y, np.imag(Z), cmap = cmap)
#     label = r"$\Im\left(\tilde{u}^+\right)$"
# case "abs":
#     pcm = ax.pcolormesh(X, Y, np.abs(Z), cmap = cmap)
#     label = r"$\left|\tilde{u}^+\right|$"
# case "angle":
#     pcm = ax.pcolormesh(X, Y, np.angle(Z), cmap = cmap)
#     label = r"$\text{arg}\left(\tilde{u}^+\right)$"
# case _:
#     raise ValueError("Invalid part argument. Choose 'real' or 'imag' or 'abs' or 'angle'.")

# fig.colorbar(pcm, ax = ax, shrink=0.5, aspect=20, label = label)

