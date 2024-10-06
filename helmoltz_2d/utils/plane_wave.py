
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

from helmoltz_2d.utils.graphics import MSColors
from helmoltz_2d.utils.graphics import prepare_points_for_pcolormesh



class PlaneWave:
    r"""
    @class PlaneWave
    @brief Represent a plane wave of form : e^{-i \vec{k} \cdot \vec{x}}.
    """
    def __init__(self, k: list|np.ndarray) -> None:
        self.k = self._validate_k(k)
        self.wave_number = np.linalg.norm(self.k)
        pass
    
    def _validate_k(self, k: list|np.ndarray) -> np.ndarray:
        """
        @brief Validate the k input.

        @param k: wave vector of the plane wave.
        @return: wave vector.
        """
        k = np.asarray(k)
        if k.shape != (2,):
            raise ValueError("k shape should be (2,) as it is a 2d vector.")
        return k
    
    def __call__(self, x: list|np.ndarray) -> np.ndarray:
        """
        @brief Evaluate the plane wave at the given points.

        @param x: points where the wave is evaluated. Should be of shape (2, N), where N is the number of points.
        @return: values of the plane wave at x.
        """
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[0] != 2:
            raise ValueError("Input points should be of shape (2, N) for a 2D wave.")
        return np.exp(-1j * np.dot(self.k, x))
    
    def radial_derivative(self, x: list|np.ndarray) -> np.ndarray:
        r"""
        @brief Evaluate \partial_r of the plane wave at x.

        @param x: points where \partial_r is evaluated. Should be of shape (2, N), where N is the number of points.
        @return: values of \partial_r of the plane wave at x.
        """
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[0] != 2:
            raise ValueError("Input points should be of shape (2, N) for a 2D wave.")
        
        r = np.linalg.norm(x, axis = 0)
        # print(r.shape)
        # print(np.dot(self.k, x).shape)
        return -1j * np.dot(self.k, x) * self(x) / r
    
    def approx(self, x: list|np.ndarray, N: int = 10) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[0] != 2:
            raise ValueError("Input points should be of shape (2, N) for a 2D wave.")
        
        r = np.linalg.norm(x, axis=0)
        cos_theta = np.dot(self.k, x) / (self.wave_number * r)
        theta = np.arccos(cos_theta)

        n_values = np.arange(-N, N + 1).reshape(-1, 1)  # Create a range of n values
        # print(f"{n_values.shape = }")
        bessel_terms = jv(n_values, self.wave_number * r)  # Compute all Bessel terms for each n
        exp_terms = np.exp(1j * n_values * theta)  # Compute all exponential terms for each n

        values = np.sum((-1j)**n_values * bessel_terms * exp_terms, axis=0)  # Sum across all n values
        return values
    
    def display(self, ax: plt.axes = None, boundaries=[[-10, 10], [-10, 10]], steps=[100, 100], field = "real"):
        """
        @brief Display the plane wave over a rectangular domain.

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
        boundaries = np.asarray(boundaries)
        lst_x = np.linspace(*boundaries[0,:], num=steps[0], endpoint=True)
        lst_y = np.linspace(*boundaries[1,:], num=steps[1], endpoint=True)
        xx, yy = np.meshgrid(lst_x, lst_y, indexing='ij')
        nodes = np.column_stack((xx.ravel(), yy.ravel()))
        
        values = self(nodes.T)
        values = self.radial_derivative(nodes.T)
        X, Y, Z = prepare_points_for_pcolormesh(*nodes.T, values)
        
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
    
    def display_approx(self, ax: plt.axes = None, boundaries=[[-10, 10], [-10, 10]], steps=[100, 100], field = "real", N: int = 10) -> None:
        """
        @brief Display the Jacobi-Anger approximation of the plane wave over a rectangular domain.

        @param ax: Optional, matplotlib axes object.
        @param boundaries: Optional, list specifying the x and y boundaries of the domain.
        @param steps: Optional, list specifying the number of steps for the mesh grid in x and y directions.
        @param part: Optional, choose 'real' or 'imag' or 'abs' or 'angle' to display the respective property of the wave.
        @param N: number of terms in the approximation.
        """
        show = False
        if ax is None:
            fig, ax = plt.subplots()
            ax.set(xlabel = r"$x$", ylabel = r"$y$", aspect = 'equal')
            show = True 
            
        boundaries = np.asarray(boundaries)
        lst_x = np.linspace(*boundaries[0,:], num=steps[0], endpoint=True)
        lst_y = np.linspace(*boundaries[1,:], num=steps[1], endpoint=True)
        xx, yy = np.meshgrid(lst_x, lst_y, indexing='ij')
        nodes = np.column_stack((xx.ravel(), yy.ravel()))
        
        values = self.approx(nodes.T, N = N)
        X, Y, Z = prepare_points_for_pcolormesh(*nodes.T, values)
        
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
    
    def display_comparison_with_approx(self, ax: plt.axes = None, boundaries=[[-10, 10], [-10, 10]], steps=[100, 100], field = "real", N: int = 10) -> None:
        """
        @brief Display the comparison between the Jacobi-Anger approximation of the plane wave and itself over a rectangular domain.

        @param ax: Optional, matplotlib axes object.
        @param boundaries: Optional, list specifying the x and y boundaries of the domain.
        @param steps: Optional, list specifying the number of steps for the mesh grid in x and y directions.
        @param part: Optional, choose 'real' or 'imag' or 'abs' or 'angle' to display the respective property of the wave.
        @param N: number of terms in the approximation.
        """
        show = False
        if ax is None:
            fig, ax = plt.subplots()
            ax.set(xlabel = r"$x$", ylabel = r"$y$", aspect = 'equal')
            show = True 
            
        boundaries = np.asarray(boundaries)
        lst_x = np.linspace(*boundaries[0,:], num=steps[0], endpoint=True)
        lst_y = np.linspace(*boundaries[1,:], num=steps[1], endpoint=True)
        xx, yy = np.meshgrid(lst_x, lst_y, indexing='ij')
        nodes = np.column_stack((xx.ravel(), yy.ravel()))
        
        error = self(nodes.T) - self.approx(nodes.T, N = N)
        # norm = np.linalg.norm(np.abs(error))
        # print(f"norm infty : {norm:.1e}")
        X, Y, Z = prepare_points_for_pcolormesh(*nodes.T, error)
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
        
        fig.colorbar(pcm, ax = ax, label = field + " of error")
        if show:
            plt.show()

if __name__ == '__main__':
    wave_number = 2
    k = wave_number * np.array([1, 0])
    # print(f"{k.shape = }")
    steps = [100, 100]
    N = 100
    field = 'real'
    u_inc = PlaneWave(k = k)
    u_inc.display(steps=steps, field=field)
    # u_inc.display_approx(N = N, steps=steps, field=field)
    # u_inc.display_comparison_with_approx(N = N, steps=steps, field=field)
    
    
    # boundaries=[[-10, 10], [-10, 10]]
    # steps=[10, 10]
    # x = omega.nodes.T
    # print(f"{x.shape = }")
    
    # r = np.linalg.norm(x, axis=0)
    # print(f"{r.shape = }")
    # u_inc.approx(x = omega.nodes.T)