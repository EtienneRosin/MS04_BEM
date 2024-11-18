"""This 
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from helmoltz_2d_BEM.geometry import Disc, Square, Ellipse  

class BSPNode:
    
    def __init__(
        self, 
        points: np.ndarray, 
        indices_x_sorted: np.ndarray, 
        indices_y_sorted: np.ndarray, 
        global_indices: np.ndarray, 
        depth: int = 0,
        parent = None) -> None:
        self.depth = depth
        self.parent = parent
        self.points = points
        self.indices_x_sorted = indices_x_sorted
        self.indices_y_sorted = indices_y_sorted
        self.global_indices = global_indices
        self.bbox = self._get_bbox(points, indices_x_sorted, indices_y_sorted)
        self.extents = np.diff(self.bbox).flatten()
        self.diameter = np.linalg.norm(self.bbox[:, 1] - self.bbox[:, 0])
        # self.diameter = self.bbox[1, 1] - self.bbox[0, 0]   # extrema of the bbox 
        self.childrens = []
        self.largest_dimension_axis = np.argmax(self.extents)
        
        
    def _get_bbox(self, points, indices_x_sorted, indices_y_sorted):
        return np.array([
            [points[indices_x_sorted[0], 0], points[indices_x_sorted[-1], 0]],
            [points[indices_y_sorted[0], 1], points[indices_y_sorted[-1], 1]]
        ])
    def split(self):
        N = self.points.shape[0] // 2
        if self.largest_dimension_axis == 0:  # Séparation selon l'axe X
            left_points = self.points[self.indices_x_sorted[:N]]
            right_points = self.points[self.indices_x_sorted[N:]]
            left_global_indices = self.global_indices[self.indices_x_sorted[:N]]
            right_global_indices = self.global_indices[self.indices_x_sorted[N:]]

            # Recalculez les indices triés pour chaque sous-ensemble
            indices_x_sorted_left = np.argsort(left_points[:, 0])
            indices_y_sorted_left = np.argsort(left_points[:, 1])
            indices_x_sorted_right = np.argsort(right_points[:, 0])
            indices_y_sorted_right = np.argsort(right_points[:, 1])

            self.childrens = [
                BSPNode(points=left_points, 
                        indices_x_sorted=indices_x_sorted_left,
                        indices_y_sorted=indices_y_sorted_left,
                        global_indices=left_global_indices,
                        depth=self.depth + 1,
                        parent = self),
                BSPNode(points=right_points, 
                        indices_x_sorted=indices_x_sorted_right,
                        indices_y_sorted=indices_y_sorted_right,
                        global_indices=right_global_indices,
                        depth=self.depth + 1,
                        parent = self)
            ]
        else:  # Séparation selon l'axe Y
            left_points = self.points[self.indices_y_sorted[:N]]
            right_points = self.points[self.indices_y_sorted[N:]]
            left_global_indices = self.global_indices[self.indices_y_sorted[:N]]
            right_global_indices = self.global_indices[self.indices_y_sorted[N:]]

            # Recalculez les indices triés pour chaque sous-ensemble
            indices_x_sorted_left = np.argsort(left_points[:, 0])
            indices_y_sorted_left = np.argsort(left_points[:, 1])
            indices_x_sorted_right = np.argsort(right_points[:, 0])
            indices_y_sorted_right = np.argsort(right_points[:, 1])

            self.childrens = [
                BSPNode(points=left_points, 
                        indices_x_sorted=indices_x_sorted_left,
                        indices_y_sorted=indices_y_sorted_left,
                        global_indices=left_global_indices,
                        depth=self.depth + 1),
                BSPNode(points=right_points, 
                        indices_x_sorted=indices_x_sorted_right,
                        indices_y_sorted=indices_y_sorted_right,
                        global_indices=right_global_indices,
                        depth=self.depth + 1)
            ]
    def is_leaf(self):
        """Vérifie si le nœud est une feuille (pas d'enfants)"""
        return len(self.childrens) == 0
    
    def display(self, max_depth, ax=None, color_map=cm.hsv):
        if ax is None:
            fig, ax = plt.subplots()
        
        color = color_map(self.depth / max_depth)  # La profondeur max est limitée à 10 pour la couleur
        color = "blue"
        rect = plt.Rectangle(
            (self.bbox[0, 0], self.bbox[1, 0]),
            self.extents[0],
            self.extents[1],
            edgecolor=color,
            facecolor="none",
            # linewidth= 5/(self.depth + 1)
        )
        ax.add_patch(rect)
        

        for children in self.childrens:
            children.display(max_depth = max_depth, ax=ax, color_map=color_map)

        ax.set_aspect('equal', 'box')


class BSPTree:
    """ Represent a 2D Binary Spatial Partitionning tree.
    
        Our method to build the tree is the following :
        - first sort the 2D points by x and by y.
        - Give the list of point and the sorted indices to a node
        - get the bounding box of these points
        - split the bounding box in the greatest direction 
        - sort the point in the other direction
        - give half the points to one sub node and the rest to the other
        - repeat
    """
    def __init__(self, points: np.ndarray) -> None:
        self.points = points
        self.indices_x_sorted = np.argsort(points[:, 0])
        self.indices_y_sorted = np.argsort(points[:, 1])
        self.root = BSPNode(points, self.indices_x_sorted, self.indices_y_sorted, np.arange(points.shape[0]))
        self.max_depth = 0
        self.indexing = None
    
    def split(self):
        self.root.split()
        
    def build_tree(self, N_leaf: int):
        """Construit récursivement l'arbre jusqu'à ce que chaque feuille ait au plus N_leaf points"""
        self._build_recursive(self.root, N_leaf)
        self.indexing = self.get_leaf_global_indices()
    
    def get_leaf_global_indices(self):
        """Récupère les indices globaux de chaque feuille de l'arbre sous forme de liste plate."""
        leaf_indices = []
        self._collect_leaf_indices(self.root, leaf_indices)
        return np.array(leaf_indices)

    def _collect_leaf_indices(self, node, leaf_indices):
        """Parcourt l'arbre et ajoute les indices globaux des feuilles."""
        if node.is_leaf():
            leaf_indices.extend(node.global_indices.tolist())  # Utilisation de extend pour aplatir la liste
        else:
            for child in node.childrens:
                self._collect_leaf_indices(child, leaf_indices)
    
    def _build_recursive(self, node: BSPNode, N_leaf: int):
        # Mettez à jour la profondeur maximale si nécessaire
        if node.depth > self.max_depth:
            self.max_depth = node.depth

        if node.points.shape[0] > N_leaf:
            node.split()
            for child in node.childrens:
                self._build_recursive(child, N_leaf)
                
    def display_indexing(self):
        # leaf_indices = self.get_leaf_global_indices()
        leaf_indices = self.indexing
        plt.scatter(leaf_indices, points[leaf_indices, 0], label = "$x$")
        plt.scatter(leaf_indices, points[leaf_indices, 1], label = "$y$")
        plt.show()
        
        plt.scatter(range(len(leaf_indices)), leaf_indices)
        plt.show()
    
    def display(self):
        fig, ax = plt.subplots()
        self.root.display(max_depth = self.max_depth, ax=ax)
        ax.scatter(*self.points.T, color="red", s=1, zorder=5)
        plt.title(f"Arbre BSP - Profondeur maximale : {self.max_depth}")
        plt.show()

# Exemple d'utilisation
if __name__ == '__main__':    
    N_e = 200
    
    a = 4.0  # semi-major axis
    b = 1.0  # semi-minor axis
    # obstacle = Ellipse(N_e=N_e, a=a, b=b)
    obstacle = Square(N_e=N_e, width=a)
    # obstacle = Disc(N_e=N_e, radius=a)
    points = obstacle.y_e
    
    tree = BSPTree(points)
    tree.build_tree(N_leaf=3)
    tree.display()
    tree.display_indexing()
    # leaf_indices = tree.get_leaf_global_indices()
    # print("Indices globaux des feuilles :", leaf_indices)
    # print(points[leaf_indices])
    
    # colors = np.array([i for i in range(len(leaf_indices))])
    # colors = 

    # Affichage des points avec couleurs selon l'indice dans leaf_indices
    
    
    
    
    # plt.scatter(*points.T, c=leaf_indices, cmap='viridis', s=2)
    # plt.title("Affichage des points colorés")
    # plt.colorbar(label="Indice des feuilles")
    # plt.show()