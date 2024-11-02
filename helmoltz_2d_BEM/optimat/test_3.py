import numpy as np

def ppca_approximation(A, epsilon):
    """
    Implémentation de l'algorithme d'approximation croisée avec pivotement partiel (PPCA).
    
    Paramètres :
    - A : Matrice d'entrée de taille (m, n)
    - epsilon : Critère de convergence (précision souhaitée)

    Retourne :
    - A_k : Matrice approximée de rang k
    - Pr : Indices des lignes pivots
    - Pc : Indices des colonnes pivots
    """
    # Dimensions de la matrice
    m, n = A.shape

    # Initialisation des ensembles de pivots et des vecteurs de mise à jour
    Pr, Pc = [], []  # Listes pour les indices des lignes et colonnes pivots
    U, V = [], []    # Listes pour stocker les vecteurs u_k et v_k
    A_k = np.zeros_like(A)  # Matrice approximée initialisée à zéro
    i_star = 0  # Premier pivot de ligne (indice 0 en Python)
    
    # Norme initiale de la matrice pour le critère d'arrêt
    norm_A = np.linalg.norm(A, 'fro')

    # for k in range(min(m, n)):
    while True:
        # Étape 1 : Ajout de la ligne pivot actuelle à Pr
        Pr.append(i_star)
        
        # Étape 2 : Calcul de la ligne résiduelle pour i_star sans former R_k explicitement
        residual_row = A[i_star, :].copy()
        for l in range(len(U)):
            residual_row -= U[l][i_star] * V[l]  # Mise à jour de la ligne résiduelle

        # Sélection de la colonne pivot j_star
        j_star = np.argmax(np.abs(residual_row))
        gamma = residual_row[j_star]

        # Vérification si le pivot est nul (convergence ou absence de solution)
        if np.abs(gamma) < epsilon:
            print("Convergence atteinte ou arrêt sans convergence.")
            break

        # Étape 3 : Ajout de la colonne pivot j_star à Pc
        Pc.append(j_star)
        
        # Calcul de la colonne résiduelle pour j_star
        residual_col = A[:, j_star].copy()
        for l in range(len(V)):
            residual_col -= V[l][j_star] * U[l] / gamma

        # Étape 4 : Mise à jour des vecteurs u_k et v_k
        u_k = residual_col / gamma
        v_k = residual_row

        # Ajout de u_k et v_k aux listes
        U.append(u_k)
        V.append(v_k)

        # Mise à jour de la matrice approximée A_k
        A_k += np.outer(u_k, v_k)

        # Critère d'arrêt basé sur la norme de Frobenius
        if np.linalg.norm(u_k) * np.linalg.norm(v_k) <= epsilon * norm_A:
            print("Convergence atteinte.")
            break

        # Sélection du prochain pivot de ligne i_star
        abs_residual_col = np.abs(residual_col)
        for idx in Pr:
            abs_residual_col[idx] = -np.inf  # Exclusion des lignes déjà sélectionnées
        i_star = np.argmax(abs_residual_col)

    return A_k, Pr, Pc

# Exemple d'utilisation
A = np.array([
    [6.5, 31, -14, -43],
    [9.1, -3, 11, 31],
    [17.6, -16, 28, 80],
    [26.2, 50, -7, -26],
])
epsilon = 1e-6
A_k, Pr, Pc = ppca_approximation(A, epsilon)

print("Matrice approximée A_k :\n", A_k)
print("Indices des lignes pivots Pr:", Pr)
print("Indices des colonnes pivots Pc:", Pc)