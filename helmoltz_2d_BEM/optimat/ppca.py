import numpy as np
from helmoltz_2d_BEM.optimat import argmax_in_subarray, generate_matrix_with_rank

def ppca(A: np.ndarray, epsilon: float, tol=1e-10):
    """Compute the Partially-Pivoted Cross Approximation of A.

    Parameters
    ----------
    A : np.ndarray
        Considered matrix.
    epsilon : float
        Assumed error.
    tol : float, optional
        Tolerance for the 0 equality check. Default is 1e-10.

    Raises
    ------
    ValueError
        If no convergence.

    Returns
    -------
    U : np.ndarray
        Considered columns.
    V : np.ndarray
        Considered rows.
    """
    P_c = set()
    P_r = set()
    m, n = A.shape
    max_iter = np.min([m, n])

    i_star = 0   # first pivot row
    k = 0
    U, V = [], []

    A_k_squared_norm_frob = 0.0
    u_k_squared_norm = 0.0
    v_k_squared_norm = 0.0

    while True:
        residual_row = A[i_star, :]
        for l in range(len(U)):
            # print(f"{k = }, {l = }")
            residual_row -= U[l][i_star] * V[l]
        
        j_star = np.argmax(np.abs(residual_row))
        delta_k = residual_row[j_star]
        
        if np.abs(delta_k) < tol:
            if len(P_c) == n - 1:
                raise ValueError(f"No convergence of the {ppca.__name__} function.")
            else:
                break
        else:
            v_k = residual_row / delta_k
            u_k = A[:, j_star]
            for l in range(len(U)):
                u_k -= V[l][j_star] * U[l]
            
            # Computation of the squared norm
            u_k_squared_norm = np.linalg.norm(u_k)**2
            v_k_squared_norm = np.linalg.norm(v_k)**2
            print(f"{k = }, {len(U) = }, {len(V) = }")
            A_k_squared_norm_frob += u_k_squared_norm * v_k_squared_norm
            A_k_squared_norm_frob += 2 * sum(np.dot(u_k, u_l) * np.dot(v_l, v_k) for u_l, v_l in zip(U, V))
            
            V.append(v_k)
            U.append(u_k)
            k += 1
        
        P_r.add(i_star)
        P_c.add(j_star)
        
        print(f"{k = }, {np.sqrt(u_k_squared_norm * v_k_squared_norm) = }, {np.sqrt(A_k_squared_norm_frob) = }")
        
        if np.sqrt(u_k_squared_norm * v_k_squared_norm) <= epsilon * np.sqrt(A_k_squared_norm_frob):
            break
        if k >= max_iter:
            raise ValueError("Max iterations reached without convergence.")
        i_star = argmax_in_subarray(a=np.abs(u_k), I=P_r)
    
    return np.array(U), np.array(V)

if __name__ == '__main__':
    A = np.array([
        [6.5, 31, -14, -43],
        [9.1, -3, 11, 31],
        [17.6, -16, 28, 80],
        [26.2, 50, -7, -26],
    ])
    
    # A = generate_matrix_with_rank(m = 50, n = 50, r = 3)
    print(f"{np.linalg.matrix_rank(A) = }")
    epsilon = 0.75
    epsilon = 1e-5
    U, V = ppca(A, epsilon)
    A_approx = sum(np.outer(U[i], V[i]) for i in range(len(U)))
    print(np.linalg.norm(A - A_approx, ord='fro'))
    
    # print("Matrice U :", U)
    # print("Matrice V :", V)
