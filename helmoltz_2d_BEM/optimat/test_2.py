import numpy as np
import sys
from helmoltz_2d_BEM.optimat import argmax_in_subarray

def ppca(A, epsilon = 1e-10):
    """
    Algorithme de décomposition avec pivots successifs en ligne et colonne.

    Paramètres:
    - A : Matrice d'entrée de taille (m, n)
    - epsilon : Critère d'arrêt pour la convergence

    Retourne:
    - P_r : Ensemble des indices des lignes pivots
    - P_c : Ensemble des indices des colonnes pivots
    - U, V : Matrices des vecteurs u_k et v_k pour reconstruire A_k
    """
    # Dimensions de la matrice
    m, n = A.shape
    

    # Initialisation des ensembles et matrices
    P_r = set()   # Ensemble des indices des lignes pivots
    P_c = set()   # Ensemble des indices des colonnes pivots
    i_star = 0   # Premier pivot ligne
    U, V = [], []  # Matrices des vecteurs u et v
    A_k = np.zeros_like(A)  # Initialisation de A_k
    sigma = set(range(m))

    while True:
        P_r.add(i_star)
        residual_row = A[i_star, :].copy()
        for l in range(len(U)):
            residual_row -= U[l][i_star] * V[l]
        
        j_star = argmax_in_subarray(a = np.abs(residual_row), I = P_c)
        gamma = residual_row[j_star]
        
        if np.abs(gamma) < epsilon:
            pass
            if len(sigma - P_r) == 0:
                if np.linalg.norm(u_k) * np.linalg.norm(v_k) < 0.5 * np.linalg.norm(A_k, 'fro'):
                    print(f"Convergence atteinte")
                    break
                else:
                    raise ValueError(f"No convergence of the {ppca.__name__} function.")
            i_star = min(sigma - P_r)
        else:
            P_c.add(j_star)
            residual_col = A[:, j_star].copy()
            for l in range(len(V)):
                residual_col -= V[l][j_star] * U[l]
                
            u_k = residual_col/gamma
            v_k = residual_row
            U.append(u_k)
            V.append(v_k)
            
            A_k += np.outer(u_k, v_k)
            
            i_star = argmax_in_subarray(a = np.abs(residual_col), I = P_r)
        # print(f"{P_r = }, {P_c = }")
        # print(f"{np.linalg.norm(u_k) * np.linalg.norm(v_k) = }, {np.linalg.norm(A_k, 'fro')}")
        if np.linalg.norm(u_k) * np.linalg.norm(v_k) < 0.75 * np.linalg.norm(A_k, 'fro'):
            print(f"Convergence atteinte")
            break

    print(A_k)
    return P_r, P_c, np.array(U), np.array(V)

# Exemple d'utilisation
A = np.array([
    [6.5, 31, -14, -43],
    [9.1, -3, 11, 31],
    [17.6, -16, 28, 80],
    [26.2, 50, -7, -26],
])
epsilon = 1e-6
P_r, P_c, U, V = ppca(A, epsilon)

print("Lignes pivots sélectionnées (P_r):", P_r)
print("Colonnes pivots sélectionnées (P_c):", P_c)
print("Matrice U :", U)
print("Matrice V :", V)