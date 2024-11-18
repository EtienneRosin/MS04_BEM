import numpy as np
import matplotlib.pyplot as plt
from helmoltz_2d_BEM.geometry import Disc, Square, Ellipse  

class BSPNode:
    def __init__(self, bbox, depth=0):
        self.depth = depth
        self.bbox = np.asarray(bbox)
        self.extents = np.diff(self.bbox).flatten()
        self.largest_dimension_axis = np.argmax(self.extents)
        self.points = []
        self.left_children = None
        self.right_children = None
        self.is_leaf = True  # Pour vérifier si un nœud est une feuille

    def contains(self, points):
        """Check if points are within the bbox."""
        return (self.bbox[0, 0] <= points[:, 0]) & (points[:, 0] <= self.bbox[0, 1]) & \
               (self.bbox[1, 0] <= points[:, 1]) & (points[:, 1] <= self.bbox[1, 1])
    
    def insert_points(self, points, N_leaf):
        """Insert points and split if the number of points exceeds N_leaf."""
        self.points = points  # Affectation initiale

        # Si le nombre de points dépasse N_leaf, divisez
        if len(self.points) > N_leaf:
            self.split(N_leaf)

    def split(self, N_leaf):
        """Split the node into two children."""
        bbox = self.bbox
        self.is_leaf = False  # Marque le nœud comme non-feuille après division
        
        if self.largest_dimension_axis == 0:
            mid = (bbox[0, 0] + bbox[0, 1]) / 2
            bbox_left = [[bbox[0, 0], mid], bbox[1]]
            bbox_right = [[mid, bbox[0, 1]], bbox[1]]
        else:
            mid = (bbox[1, 0] + bbox[1, 1]) / 2
            bbox_left = [bbox[0], [bbox[1, 0], mid]]
            bbox_right = [bbox[0], [mid, bbox[1, 1]]]

        # Création des enfants
        self.left_children = BSPNode(bbox=bbox_left, depth=self.depth + 1)
        self.right_children = BSPNode(bbox=bbox_right, depth=self.depth + 1)

        # Division des points entre les enfants
        left_mask = self.left_children.contains(self.points)
        right_mask = ~left_mask
        left_points = self.points[left_mask]
        right_points = self.points[right_mask]
        
        # Insertion des points dans les enfants
        self.left_children.insert_points(left_points, N_leaf)
        self.right_children.insert_points(right_points, N_leaf)
        
        # Nettoyage des points au niveau du parent après insertion dans les enfants
        self.points = None

    def get_ordered_points(self):
        """Get points in spatial order (preorder traversal) for new indexing."""
        ordered_points = []
        if self.is_leaf:
            ordered_points.extend(self.points)
        else:
            if self.left_children:
                ordered_points.extend(self.left_children.get_ordered_points())
            if self.right_children:
                ordered_points.extend(self.right_children.get_ordered_points())
        return ordered_points

    def display(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        # Trace le rectangle du nœud actuel
        rect = plt.Rectangle(
            (self.bbox[0, 0], self.bbox[1, 0]),
            self.extents[0],
            self.extents[1],
            edgecolor="blue",
            facecolor="none",
            linewidth=0.5
        )
        ax.add_patch(rect)

        # Affiche les enfants de manière récursive
        if self.left_children:
            self.left_children.display(ax)
        if self.right_children:
            self.right_children.display(ax)

        ax.set_aspect('equal', 'box')

if __name__ == '__main__':
    
    N_e = 4000
    a = 4.0  # semi-major axis
    b = 1.0  # semi-minor axis
    
    obstacle = Ellipse(N_e=N_e, a=a, b=b)
    # obstacle = Square(N_e=N_e, width=a)
    points = obstacle.y_e

    bbox = 1.01 * np.array([[points[:, 0].min(), points[:, 0].max()], 
                            [points[:, 1].min(), points[:, 1].max()]])
    
    root = BSPNode(bbox=bbox)
    N_leaf = 10
    root.insert_points(points, N_leaf)

    # Récupération des points ordonnés
    ordered_points = np.array(root.get_ordered_points())

    # Affichage de l'arbre et des points
    fig, ax = plt.subplots()
    root.display(ax=ax)
    ax.scatter(ordered_points[:, 0], ordered_points[:, 1], color='red', s=2, zorder=5)
    plt.show()