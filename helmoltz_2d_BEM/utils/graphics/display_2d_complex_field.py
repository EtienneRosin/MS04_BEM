import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import cmasher as cmr
# plt.style.use('science')

from helmoltz_2d_BEM.utils import prepare_points_for_pcolormesh

def display_2d_complex_field(x: np.ndarray, y: np.ndarray, u: np.ndarray, field = "real", save_name: str = None):
        """
        @brief Display a 2D complex field.
        
        @param x: x-direction positions
        @param y: y-direction positions
        @param u: complex field to display
        @param field: (Optional) choose 'real' or 'imag' or 'abs' or 'angle' to display the respective property of the solution.
        @param save_name: (Optional) save name of the figure if provided 
        """
        with plt.style.context('science' if save_name else 'default'):
            fig, ax = plt.subplots()
            ax.set(xlabel = r"$x$", ylabel = r"$y$", aspect = 'equal')
            
            X, Y, Z = prepare_points_for_pcolormesh(x, y, u)
            cmap = cmr.lavender
            match field:
                case "real":
                    pcm = ax.pcolormesh(X, Y, np.real(Z), cmap = cmap)
                    label = r"$\Re\left(\tilde{u}^+\right)$"
                case "imag":
                    pcm = ax.pcolormesh(X, Y, np.imag(Z), cmap = cmap)
                    label = r"$\Im\left(\tilde{u}^+\right)$"
                case "abs":
                    pcm = ax.pcolormesh(X, Y, np.abs(Z), cmap = cmap)
                    label = r"$\left|\tilde{u}^+\right|$"
                case "angle":
                    pcm = ax.pcolormesh(X, Y, np.angle(Z), cmap = cmap)
                    label = r"$\text{arg}\left(\tilde{u}^+\right)$"
                case _:
                    raise ValueError("Invalid part argument. Choose 'real' or 'imag' or 'abs' or 'angle'.")
            
            fig.colorbar(pcm, ax = ax, shrink=0.5, aspect=20, label = label)
            if save_name:
                fig.savefig(f"{save_name}_{field}.pdf")
            plt.show()
            
if __name__ == '__main__':
    save_name = None
    if save_name :
        print("iibb")