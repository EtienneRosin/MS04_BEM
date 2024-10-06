
from helmoltz_2d.geometry import Disc, RectangularDomainWithDisc
from helmoltz_2d.utils import PlaneWave
from helmoltz_2d.disc_case import ExactSolution
from helmoltz_2d.utils.quadratures.gauss_legendre_2_points import quadrature_points, quadrature_weights
from helmoltz_2d.utils.graphics import prepare_points_for_pcolormesh, MSColors

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, hankel1

import scienceplots
import cmasher as cmr
plt.style.use('science')

class IntegralRepresentationWithKnownSolution:
    r"""
    @class IntegralRepresentationWithKnownSolution
    @brief Represent the integral representation of the 2D Helmoltz equation with Dirichlet condition of a plane wave ^{-i \vec{k} \cdot \vec{x}}.
    """
    def __init__(self, domain_boundaries: list|np.ndarray, domain_steps: int|list|np.ndarray, k: list|np.ndarray, a: float, N_e: int, N: int) -> None:
        """
        @brief Constructor.
        
        @param boundaries: boundaries of the domain.
        @param steps: step size in each direction.
        @param k: wave vector of the initial plane wave.
        @param a: radius of the disc.
        @param N_e: number of elements of the disc' boundary discretization.
        @param N: number of terms of the approximation (of the exaction solution).
        """
        self.domain = RectangularDomainWithDisc(boundaries=domain_boundaries, steps=domain_steps, disc=Disc(N_e=N_e, radius=a))
        self.u_inc = PlaneWave(k = k)
        self.u_plus = ExactSolution(k = k, a = a, N = N)
        self.a = a
        
        self.U = None
        
    def p(self, x: list|np.ndarray) -> np.ndarray:
        """
        @brief Evaluate the trace of the normal derivative

        @param x: points where the wave is evaluated. Should be of shape (2, N), where N is the number of points.
        @return: values of trace of the normal derivative at x.
        """
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[0] != 2:
            raise ValueError("Input points should be of shape (2, N) for a 2D wave.")
        return self.u_inc.radial_derivative(x) + self.u_plus.radial_derivative(x)
    
    def display_p(self) -> None:
        """
        @brief Display the trace of the normal derivative on the disc boundary
        """
        lst_theta = np.linspace(start=-np.pi,stop=np.pi, num=1000, endpoint=True)
        R, THETA = np.meshgrid(self.a, lst_theta, indexing='ij')
        polar_nodes = np.column_stack((R.ravel(), THETA.ravel()))
        XX, YY = self.a * np.cos(THETA), self.a * np.sin(THETA)
        nodes = np.column_stack((XX.ravel(), YY.ravel()))
        values = self.p(nodes.T)
        
        fig = plt.figure()
        ax = fig.add_subplot()
        
        ax.plot(lst_theta, values.real, label = r"$\Re (p)$", c = MSColors.LIGHT_BLUE)
        ax.plot(lst_theta, values.imag, label = r"$\Im (p)$", c = MSColors.ORANGE)
        ax.set(xlabel = r"$\theta$")
        ax.legend()
        # fig.savefig("trace.pdf")
        plt.show()
        
    def construct_vector_p(self) -> np.ndarray:
        r"""
        @brief Construct the vector P such that (P)_e = p(y_e_m)
        """
        y_e_m = self.domain.disc.middle_nodes
        # print(f"{y_e_m.shape = }")
        # print(self.p(y_e_m.T).shape)
        return self.p(y_e_m.T)
    
    def construct_matrix_G(self) -> np.ndarray:
        r"""
        @brief Construct the matrix G such that (G)_{ei} = ||y_e^m|| \sum_{q = 1}^2 \omega^q  G(x_i, (x^q+1)y_e^m + y_e)
            with G(x, y) = i/4 H_0^(1)(k||x - y||)
        """
        y_e_m = self.domain.disc.middle_nodes
        y_e = self.domain.disc.elements_first_node
        x_i = self.domain.nodes
        k = self.u_inc.wave_number
        
        # fig, ax = plt.subplots()
        # ax.scatter(*y_e.T, label = "boundary", s = 5)
        # for omega_q, x_q in zip(quadrature_weights, quadrature_points):
        #     ax.scatter(*((x_q + 1)*y_e_m + y_e).T, label = "poi", s = 5)
        
        
        # # ax.scatter(*y_e_m.T, label = "middle", s = 5)
        # # ax.scatter(*x_i.T, label = "domain", s = 1)
        # ax.set(aspect = "equal")
        # ax.legend()
        # plt.show()
        
        
        
        G = np.zeros((x_i.shape[0], y_e.shape[0]), dtype=complex)
        # for omega_q, x_q in zip(quadrature_weights, quadrature_points):
        #     r = np.linalg.norm(
        #         x_i[:, np.newaxis, :] - ((x_q + 1)*y_e_m + y_e)[np.newaxis, :, :], 
        #         axis = 2) # norm of \vec{x}_i - ((x_q + 1) \vec{y}_e^m + \vec{y}_e)
        #     G += omega_q * hankel1(0, k * r)
        for omega_q, x_q in zip(quadrature_weights, quadrature_points):
            r = np.linalg.norm(
                x_i[:, np.newaxis, :] - ((x_q + 1) * y_e_m + y_e)[np.newaxis, :, :], 
                axis=2)  # norm of \vec{x}_i - ((x_q + 1) \vec{y}_e^m + \vec{y}_e)
            # r = np.linalg.norm(
            #     x_i[:, np.newaxis, :] - ((x_q + 1) * y_e_m )[np.newaxis, :, :], 
            #     axis=2)  
            # Ajout de vérification pour r
            if np.any(r == 0):
                print("Avertissement : r contient des valeurs nulles, ce qui peut conduire à des problèmes de calcul.")
            
            G += omega_q * np.linalg.norm(y_e_m, axis = 1) * (1j / 4) * hankel1(0, k * r)
        return G
    
    def display(self, field = "real", save = False):
        """
        @brief Display the approximate solution over the domain domain.
        
        @param field: Optional, choose 'real' or 'imag' or 'abs' or 'angle' to display the respective property of the wave.
        """
        fig, ax = plt.subplots()
        ax.set(xlabel = r"$x$", ylabel = r"$y$", aspect = 'equal')
        
        if self.U is None:
            self.U = self.construct_matrix_G() @ self.construct_vector_p()
        
        X, Y, Z = prepare_points_for_pcolormesh(*self.domain.nodes.T, self.U)
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
        
        # fig.colorbar(pcm, ax = ax, label = label)
        fig.colorbar(pcm, ax = ax, shrink=0.5, aspect=20, label = label)
        if save:
            fig.savefig("u_tilde_" + field + ".pdf")
        plt.show()
    
    def display_exact_solution(self, field = "real", save = False):
        """
        @brief Display the exact solution over the domain domain.
        
        @param field: Optional, choose 'real' or 'imag' or 'abs' or 'angle' to display the respective property of the wave.
        """
        fig, ax = plt.subplots()
        ax.set(xlabel = r"$x$", ylabel = r"$y$", aspect = 'equal')
        
        X, Y, Z = prepare_points_for_pcolormesh(*self.domain.nodes.T, self.u_plus(self.domain.nodes.T))
        
        cmap = cmr.lavender
        match field:
            case "real":
                pcm = ax.pcolormesh(X, Y, np.real(Z), cmap = cmap)
                label = r"$\Re\left(u^+\right)$"
            case "imag":
                pcm = ax.pcolormesh(X, Y, np.imag(Z), cmap = cmap)
                label = r"$\Im\left(u^+\right)$"
            case "abs":
                pcm = ax.pcolormesh(X, Y, np.abs(Z), cmap = cmap)
                label = r"$\left|u^+\right|$"
            case "angle":
                pcm = ax.pcolormesh(X, Y, np.angle(Z), cmap = cmap)
                label = r"$\text{arg}\left(u^+\right)$"
            case _:
                raise ValueError("Invalid part argument. Choose 'real' or 'imag' or 'abs' or 'angle'.")
        
        # fig.colorbar(pcm, ax = ax, label = label)
        fig.colorbar(pcm, ax = ax, shrink=0.5, aspect=20, label = label)
        if save:
            fig.savefig("u_" + field + ".pdf")
        # fig.colorbar(pcm, ax = ax)
        plt.show()
        
    def display_comparison(self, field = "real", save = False):
        """
        @brief Display the comparison between the approximate and exact solutions over the domain.
        
        @param field: Optional, choose 'real' or 'imag' or 'abs' or 'angle' to display the respective property of the wave.
        """
        fig, ax = plt.subplots()
        ax.set(xlabel = r"$x$", ylabel = r"$y$", aspect = 'equal')
        if self.U is None:
            self.U = self.construct_matrix_G() @ self.construct_vector_p()
        
        X, Y, Z_exact = prepare_points_for_pcolormesh(*self.domain.nodes.T, self.u_plus(self.domain.nodes.T))
        X, Y, Z = prepare_points_for_pcolormesh(*self.domain.nodes.T, self.U)
        
        error = Z_exact - Z
        
        
        cmap = cmr.lavender
        match field:
            case "real":
                pcm = ax.pcolormesh(X, Y, np.real(error), cmap = cmap)
                label = r"$\Re\left(u^+ - \tilde{u}^+\right)$"
            case "imag":
                pcm = ax.pcolormesh(X, Y, np.imag(error), cmap = cmap)
                label = r"$\Im\left(u^+ - \tilde{u}^+\right)$"
            case "abs":
                pcm = ax.pcolormesh(X, Y, np.abs(error), cmap = cmap)
                label = r"$\left|u^+ - \tilde{u}^+\right|$"
            case "angle":
                pcm = ax.pcolormesh(X, Y, np.angle(error), cmap = cmap)
                label = r"$\text{arg}\left(u^+ - \tilde{u}^+\right)$"
            case _:
                raise ValueError("Invalid part argument. Choose 'real' or 'imag' or 'abs' or 'angle'.")
        
        # fig.colorbar(pcm, ax = ax, label = label)
        fig.colorbar(pcm, ax = ax, shrink=0.5, aspect=20, label = label)
        if save:
            fig.savefig("error_" + field + ".pdf")
        
        
        plt.show()
        
if __name__ == '__main__':
    # Domain ------------------------------------
    boundaries = [[-4, 8], [-3, 3]]
    # steps = 1 * np.array([120, 30])
    steps = 10 * np.array([12, 6])
    
    # Disc --------------------------------------
    N_e = 60
    a = 1
    
    # Plane wave --------------------------------
    wave_number = 2*np.pi
    k = wave_number * np.array([1, 1])
    
    # Exact solution ----------------------------
    N = 100
    
    # Integral representation --------------------
    integral_repr = IntegralRepresentationWithKnownSolution(domain_boundaries=boundaries, domain_steps=steps, k = k, a = a, N_e=N_e, N=N)
    
    
    # Tests -------------------------------------
    # integral_repr.display_slice_theta_zero()
    # integral_repr.display_p()
    # integral_repr.construct_matrix_G()
    
    save = False
    field='angle'
    integral_repr.display(field=field, save = save)
    integral_repr.display_comparison(field=field, save = save)
    # for field in ['real', 'imag', 'abs', 'angle']:
    #     integral_repr.display(field=field, save = save)
    #     integral_repr.display_exact_solution(field=field, save = save)
    #     integral_repr.display_comparison(field=field, save = save)