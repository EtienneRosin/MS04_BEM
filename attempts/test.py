import numpy as np

# Exemple de tableau de points (N, 2) avec les colonnes x et y
points = np.array([
    [1, 2],
    [3, 1],
    [1, 1],
    [2, 3],
    [3, 2]
])

# Utiliser numpy.lexsort pour trier par y d'abord, puis par x.
# On passe d'abord l'index de la colonne secondaire (y), puis l'index de la colonne principale (x).
sorted_indices = np.lexsort((points[:, 1], points[:, 0]))

# Obtenir les points triés en utilisant les indices obtenus
sorted_points = points[sorted_indices]

print("Points triés par x puis par y :")
print(sorted_points)