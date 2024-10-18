import csv
import numpy as np
from helmoltz_2d_BEM.geometry.domains import RectangularDomainWithObstacle
from helmoltz_2d_BEM.utils import PlaneWave
from helmoltz_2d_BEM.utils.graphics.display_2d_complex_field import display_2d_complex_field
# from helmoltz_2d_BEM.i_e import IntegralRepresentation
from helmoltz_2d_BEM.integral_equation import IntegralEquation
from helmoltz_2d_BEM.geometry.obstacles import Disc
from helmoltz_2d_BEM.circle_obstacle_case import ExactSolution, ExactNormaTrace


# polar_nodes = self.obstacle.polar_nodes
# lst_theta = polar_nodes.T[1].T[self.disc.Gamma_e].mean(axis = 1)

# Domain ------------------------------------
boundaries = [[-4, 8], [-3, 3]]
steps = 10 * np.array([12, 6])

# Disc --------------------------------------
a = 1
lst_N_e = np.arange(start=100, step=100, stop=2500)

folder: str = "helmoltz_2d_BEM/validation/3_IE"
filename: str = "3_IE_validation_measurement.csv"

if __name__ == '__main__':
    with open(f'{folder}/{filename}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["N_e", "error", "A_assembly_time", "solve_time"])
        
        for N_e in lst_N_e:
            disc = Disc(N_e=N_e, radius=a)
            # Plane wave --------------------------------
            wave_number = 2*np.pi
            k = wave_number * np.array([1, 1])

            # Exact solution ----------------------------
            N = 100
            u_inc = PlaneWave(k=k)
            u_plus = ExactSolution(k=k, a=a, N=N)
            p = ExactNormaTrace(disc=disc, u_inc=u_inc, u_plus=u_plus)
            p_exact = p(disc.y_e_m.T)
            

            # Integral representation -------------------
            i_e = IntegralEquation(u_inc = PlaneWave(k = k), obstacle = disc)
            p = i_e.solve()

            # Calculate error -----------------------------
            Z = (p_exact - p) / np.linalg.norm(p_exact)
            norm_Z = np.linalg.norm(Z)

            # Afficher la norme de Z
            print(f"{N_e = }, {norm_Z = }, {i_e.A_assembly_time = }, {i_e.solve_time = }")

            # Enregistrer N_e et norm(Z) dans le fichier CSV
            writer.writerow([N_e, norm_Z, i_e.A_assembly_time, i_e.solve_time])

        print(f"Results are saved in {folder}/{filename}.")