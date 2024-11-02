import numpy as np

def decomposition_avec_pivot_partiel(A, epsilon):
    """
    Décompose la matrice A en utilisant un processus itératif avec pivot partiel.
    
    Paramètres:
    - A: Matrice d'entrée de taille (m, n)
    - epsilon: Critère d'arrêt pour la norme de Frobenius
    
    Retourne:
    - I: Ensemble des indices des lignes
    - J: Ensemble des indices des colonnes
    """
    # Initialisation des variables
    m, n = A.shape
    R_prev = A.copy()
    I, J = set(), set()
    k = 1
    i_star = 0  # i* commence à 1 (indice Python commence à 0)
    norm_A_frobenius = np.linalg.norm(A, 'fro')  # Norme de Frobenius initiale de A
    
    # Norme Frobenius initiale pour A_k
    norm_Ak_frobenius_sq = norm_A_frobenius**2

    while True:
        # Étape 3 : Trouver j* tel que la valeur absolue |R_prev(i*, j)| soit maximale
        j_star = np.argmax(np.abs(R_prev[i_star, :]))
        
        # Étape 4 : Calculer δk
        delta_k = R_prev[i_star, j_star]
        
        if delta_k == 0:
            if len(I) == min(m, n) - 1:
                print("Arrêt : tous les pivots sont trouvés.")
                break
        else:
            # Étape 10 : Calcul de uk et vk
            u_k = R_prev[:, j_star]
            v_k = R_prev[i_star, :] / delta_k
            
            # Étape 11 : Mise à jour de R_k implicitement en utilisant uk et vk
            # (Rk n'est pas explicitement formée)
            
            # Étape 15 : Calcul de i* pour la prochaine itération
            abs_uk = np.abs(u_k)
            abs_uk[list(I)] = -np.inf  # Ignorer les indices déjà dans I
            i_star = np.argmax(abs_uk)
            
            # Mise à jour de la norme de Frobenius de A_k
            norm_u_k = np.linalg.norm(u_k)
            norm_v_k = np.linalg.norm(v_k)
            
            norm_Ak_frobenius_sq += 2 * sum(u_k.T[:k-1] @ v_k[:k-1]) + (norm_u_k * norm_v_k)**2
            
            # Vérifier le critère d'arrêt
            if norm_u_k * norm_v_k <= epsilon * np.sqrt(norm_Ak_frobenius_sq):
                print("Critère d'arrêt satisfait.")
                break

            # Étape 12 : Incrémenter k
            k += 1

            # Étape 14 : Mise à jour des ensembles I et J
            I.add(i_star)
            J.add(j_star)

    return I, J

# Exemple d'utilisation
# A = np.random.rand(5, 5)  # Exemple de matrice aléatoire 5x5
A = np.array([
        [6.5, 31, -14, -43],
        [9.1, -3, 11, 31],
        [17.6, -16, 28, 80],
        [26.2, 50, -7, -26],
    ])
epsilon = 1e-6
I, J = decomposition_avec_pivot_partiel(A, epsilon)

print("Indices de lignes sélectionnés (I):", I)
print("Indices de colonnes sélectionnés (J):", J)