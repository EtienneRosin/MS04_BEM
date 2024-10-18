import csv
import numpy as np
from helmoltz_2d_BEM.geometry.domains import RectangularDomainWithObstacle
from helmoltz_2d_BEM.utils import PlaneWave
from helmoltz_2d_BEM.utils.graphics.display_2d_complex_field import display_2d_complex_field
from helmoltz_2d_BEM.integral_representation import IntegralRepresentation
from helmoltz_2d_BEM.geometry.obstacles import Disc
from helmoltz_2d_BEM.circle_obstacle_case import ExactSolution, ExactNormaTrace

# Domain ------------------------------------
boundaries = [[-4, 8], [-3, 3]]
steps = 10 * np.array([12, 6])

# Disc --------------------------------------
a = 1
lst_N_e = np.arange(start=100, step=50, stop=1000)

folder: str = "helmoltz_2d_BEM/validation/2_IR"
filename: str = "2_IR_validation_measurement.csv"

if __name__ == '__main__':
    with open(f'{folder}/{filename}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["N_e", "error", "G_assembly_time", "solve_time"])
        
        for N_e in lst_N_e:
            disc = Disc(N_e=N_e, radius=a)
            Omega = RectangularDomainWithObstacle(boundaries=boundaries, steps=steps, obstacle=disc)

            # Plane wave --------------------------------
            wave_number = 2*np.pi
            k = wave_number * np.array([1, 1])

            # Exact solution ----------------------------
            N = 100
            u_inc = PlaneWave(k=k)
            u_plus = ExactSolution(k=k, a=a, N=N)
            U_exact = u_plus(Omega.nodes.T)
            p = ExactNormaTrace(disc=disc, u_inc=u_inc, u_plus=u_plus)

            # Integral representation -------------------
            integral_representation = IntegralRepresentation(domain=Omega, u_inc=u_inc, normal_trace=p(disc.y_e_m.T))
            U = integral_representation.solve()

            # Calculate error -----------------------------
            Z = (U_exact - U) / np.linalg.norm(U_exact)
            norm_Z = np.linalg.norm(Z)
            print(f"{N_e = }, {norm_Z = }, {integral_representation.G_assembly_time = }, {integral_representation.solve_time = }")
            writer.writerow([N_e, norm_Z, integral_representation.G_assembly_time, integral_representation.solve_time])

        print(f"Results are saved in {folder}/{filename}.")