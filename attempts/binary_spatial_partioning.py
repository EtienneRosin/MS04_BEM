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
    
    def contains(self, point):
        return np.all((self.bbox[:, 0] <= point) & (point <= self.bbox[:, 1]))
    
    def insert_points(self, points, N_leaf):
        """Insert points and split if the number of points exceeds N_leaf."""
        self.points.extend(points)
        if len(self.points) > N_leaf:
            self.split()
            left_points = [p for p in self.points if self.left_children.contains(p)]
            right_points = [p for p in self.points if self.right_children.contains(p)]
            self.left_children.insert_points(left_points, N_leaf)
            self.right_children.insert_points(right_points, N_leaf)
            self.points = []  # Efface les points de ce nœud après la division

    def split(self):
        if self.left_children is not None or self.right_children is not None:
            return  # Évite de diviser si le nœud est déjà divisé
        
        bbox = self.bbox
        if self.largest_dimension_axis == 0:
            mid_x = bbox[0, 0] + self.extents[0] / 2
            bbox_left = [[bbox[0, 0], mid_x], bbox[1]]
            bbox_right = [[mid_x, bbox[0, 1]], bbox[1]]
        elif self.largest_dimension_axis == 1:
            mid_y = bbox[1, 0] + self.extents[1] / 2
            bbox_left = [bbox[0], [bbox[1, 0], mid_y]]
            bbox_right = [bbox[0], [mid_y, bbox[1, 1]]]
        else:
            raise ValueError("Invalid axis for splitting.")

        self.left_children = BSPNode(bbox=bbox_left, depth=self.depth + 1)
        self.right_children = BSPNode(bbox=bbox_right, depth=self.depth + 1)

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
        # ax.set_xlim(self.bbox[0])
        # ax.set_ylim(self.bbox[1])

if __name__ == '__main__':
    
    # np.random.seed(0)
    # N = 10000
    # points = np.random.uniform(-5, 5, (N, 2))
    # bbox = [[-5, 5], [-5, 5]]
    
    N_e = 4000
    
    a = 4.0  # semi-major axis
    b = 1.0  # semi-minor axis
    obstacle = Ellipse(N_e=N_e, a=a, b=b)
    # obstacle = Disc(N_e=N_e, radius=a)
    # obstacle = Square(N_e=N_e, width=a)
    
    
    
    
    points = obstacle.y_e
    bbox = 1.01*np.array([[points[:, 0].min(), points[:, 0].max()], [points[:, 1].min(), points[:, 1].max()]])
    
    root = BSPNode(bbox=bbox)

    N_leaf = 2
    root.insert_points(points, N_leaf)

    fig, ax = plt.subplots()
    root.display(ax=ax)
    ax.scatter(points[:, 0], points[:, 1], color='red', s=2, zorder=5)
    plt.show()