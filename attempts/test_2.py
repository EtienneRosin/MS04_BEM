import numpy as np
import matplotlib.pyplot as plt
from helmoltz_2d_BEM.geometry import Disc, Square, Ellipse  

class KDTreeNode:
    def __init__(self, points, depth=0, bounds=None):
        self.points = points
        self.left = None
        self.right = None
        self.axis = depth % 2  # 0 pour x, 1 pour y
        self.bounds = bounds  # Limites de l'espace (xmin, xmax, ymin, ymax)

def build_kdtree(points, N_leaf, depth=0, bounds=None):
    if len(points) <= N_leaf:
        return KDTreeNode(points, depth, bounds)
    
    axis = depth % 2
    points = points[points[:, axis].argsort()]
    median_idx = len(points) // 2

    median_point = points[median_idx]
    left_bounds = list(bounds)
    right_bounds = list(bounds)
    
    if axis == 0:  # Division selon l'axe x
        left_bounds[1] = median_point[0]  # Mise à jour de xmax pour le sous-espace gauche
        right_bounds[0] = median_point[0]  # Mise à jour de xmin pour le sous-espace droit
    else:  # Division selon l'axe y
        left_bounds[3] = median_point[1]  # Mise à jour de ymax pour le sous-espace gauche
        right_bounds[2] = median_point[1]  # Mise à jour de ymin pour le sous-espace droit
    
    node = KDTreeNode(points[median_idx:median_idx+1], depth, bounds)
    node.left = build_kdtree(points[:median_idx], N_leaf, depth + 1, left_bounds)
    node.right = build_kdtree(points[median_idx + 1:], N_leaf, depth + 1, right_bounds)
    
    return node

def plot_kdtree(node, ax):
    if node is None:
        return
    
    # Si c'est une feuille, dessine le rectangle correspondant
    if node.left is None and node.right is None:
        x_min, x_max, y_min, y_max = node.bounds
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                             edgecolor='black', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.plot(node.points[:, 0], node.points[:, 1], 'bo', markersize = 1)  # Points en bleu dans la feuille
        return
    
    # Trace récursivement les sous-arbres
    plot_kdtree(node.left, ax)
    plot_kdtree(node.right, ax)

# Génère des points et construit l'arbre
N_leaf = 10
# points = np.random.rand(10000, 2)  # Points aléatoires
N_e = 200
a = 4.0  # semi-major axis
b = 1.0  #
obstacle = Ellipse(N_e=N_e, a=a, b=b)
# obstacle = Disc(N_e=N_e, radius=a)
points = obstacle.y_e
    
bounds = [0, 1, 0, 1]  # Limites de l'espace initial [xmin, xmax, ymin, ymax]
kd_tree = build_kdtree(points, N_leaf, 0, bounds)

# Affiche les boîtes et les points
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(points[:, 0], points[:, 1], c='red', s=1)  # Tous les points en rouge
plot_kdtree(kd_tree, ax)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()