import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from helmoltz_2d_BEM.geometry import Disc, Square, Ellipse  
import cmasher as cmr

class BSPNode:
    """
    Represents a node in a 2D Binary Space Partitioning (BSP) tree, storing
    a subset of points with associated bounding box information and recursive subdivisions.
    """
    
    def __init__(
        self, 
        points: np.ndarray, 
        indices_x_sorted: np.ndarray, 
        indices_y_sorted: np.ndarray, 
        global_indices: np.ndarray, 
        depth: int = 0,
        parent=None
    ) -> None:
        """
        Initialize a BSP node with point data and sorted indices.
        
        Parameters:
        - points: np.ndarray -> Points contained within this node.
        - indices_x_sorted: np.ndarray -> Indices of points sorted by x-coordinate.
        - indices_y_sorted: np.ndarray -> Indices of points sorted by y-coordinate.
        - global_indices: np.ndarray -> Global indices of points for external reference.
        - depth: int -> Depth level of this node in the BSP tree.
        - parent: BSPNode or None -> Reference to parent node (None for the root).
        """
        self.depth = depth
        self.parent = parent
        self.points = points
        self.indices_x_sorted = indices_x_sorted
        self.indices_y_sorted = indices_y_sorted
        self.global_indices = global_indices
        self.bbox = self._calculate_bbox(points, indices_x_sorted, indices_y_sorted)
        self.childrens = []
        self._extents = np.diff(self.bbox).flatten()
        self.largest_dimension_axis = np.argmax(self.extents)

    @staticmethod
    def _calculate_bbox(points, indices_x_sorted, indices_y_sorted):
        """Calculate bounding box for the points in this node."""
        return np.array([
            [points[indices_x_sorted[0], 0], points[indices_x_sorted[-1], 0]],
            [points[indices_y_sorted[0], 1], points[indices_y_sorted[-1], 1]]
        ])

    @property
    def extents(self):
        """Calculate the extents (width and height) of the bounding box."""
        return self._extents

    @property
    def diameter(self):
        """Calculate the diameter (diagonal length) of the bounding box."""
        return np.linalg.norm(self.bbox[:, 1] - self.bbox[:, 0])

    def split(self):
        """
        Split the node along its largest dimension and create two children nodes.
        
        The split is performed along the largest dimension of the bounding box. Each
        child node will contain half of the points.
        """
        N = self.points.shape[0] // 2
        # N = (self.points.shape[0]+1) // 2
        if self.largest_dimension_axis == 0:  # Split along X axis
            left_indices, right_indices = self.indices_x_sorted[:N], self.indices_x_sorted[N:]
            # indices_x_sorted_left = left_indices
            # indices_x_sorted_right = right_indices
            # indices_y_sorted_left = np.argsort(self.points[left_indices, 1])
            # indices_y_sorted_right = np.argsort(self.points[right_indices, 1])
            # print(f"Cutting in the x direction ------------------------")
            # print(f"{len(left_indices) = }, {len(indices_x_sorted_left) = }, {len(indices_y_sorted_left) = }")
            # print(f"{len(right_indices) = }, {len(indices_x_sorted_right) = }, {len(indices_y_sorted_right) = }")
            # print(np.argsort(self.points[right_indices, 0]))
        else:  # Split along Y axis
            left_indices, right_indices = self.indices_y_sorted[:N], self.indices_y_sorted[N:]
            # indices_y_sorted_left = left_indices
            # indices_y_sorted_right = right_indices
            # indices_x_sorted_left = np.argsort(self.points[left_indices, 0])
            # indices_x_sorted_right = np.argsort(self.points[right_indices, 0])
            # print(f"Cutting in the y direction ------------------------")
            # print(f"{len(left_indices) = }, {len(indices_x_sorted_left) = }, {len(indices_y_sorted_left) = }")
            # print(f"{len(right_indices) = }, {len(indices_x_sorted_right) = }, {len(indices_y_sorted_right) = }")

        # Create child nodes with sorted indices in each dimension
        self.childrens = [
            BSPNode(
                points=self.points[left_indices], 
                indices_x_sorted=np.argsort(self.points[left_indices, 0]),
                indices_y_sorted=np.argsort(self.points[left_indices, 1]),
                global_indices=self.global_indices[left_indices],
                depth=self.depth + 1,
                parent=self
            ),
            BSPNode(
                points=self.points[right_indices], 
                indices_x_sorted=np.argsort(self.points[right_indices, 0]),
                indices_y_sorted=np.argsort(self.points[right_indices, 1]),
                global_indices=self.global_indices[right_indices],
                depth=self.depth + 1,
                parent=self
            )
        ]
        

    def is_leaf(self):
        """Return True if the node is a leaf (has no children)."""
        return len(self.childrens) == 0
    
    def display(self, max_depth, ax=None, color_map=cm.hsv, alpha = 0.75):
        """
        Recursively display the node's bounding box and those of its children.

        Parameters:
        - max_depth: int -> Maximum depth of the tree for color scaling.
        - ax: matplotlib.axes.Axes or None -> Axes to plot on (creates new Axes if None).
        - color_map: matplotlib.cm -> Colormap to represent depth.
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        color = color_map(self.depth / max_depth)
        rect = plt.Rectangle(
            (self.bbox[0, 0], self.bbox[1, 0]),
            self.extents[0],
            self.extents[1],
            edgecolor=color,
            facecolor = color, 
            alpha = alpha
            # facecolor="none"
        )
        ax.add_patch(rect)
        
        for child in self.childrens:
            child.display(max_depth=max_depth, ax=ax, color_map=color_map, alpha=alpha)

        ax.set_aspect('equal', 'box')


class BSPTree:
    """ Represents a 2D Binary Spatial Partitioning tree."""
    
    def __init__(self, points: np.ndarray) -> None:
        """
        Initialize the BSP tree with a set of points.
        
        Parameters:
        - points: np.ndarray -> Array of 2D points.
        """
        self.points = points
        self.indices_x_sorted = np.argsort(points[:, 0])
        self.indices_y_sorted = np.argsort(points[:, 1])
        self.root = BSPNode(points, self.indices_x_sorted, self.indices_y_sorted, np.arange(points.shape[0]))
        self.max_depth = 0
        self.indexing = None
    
    def build_tree(self, N_leaf: int):
        """
        Recursively build the BSP tree until each leaf node has at most N_leaf points.
        
        Parameters:
        - N_leaf: int -> Maximum number of points allowed in each leaf node.
        """
        self._build_recursive(self.root, N_leaf)
        self.indexing = self.get_leaf_global_indices()
    
    def get_leaf_global_indices(self):
        """Return global indices of all points in leaf nodes as a flattened list."""
        leaf_indices = []
        self._collect_leaf_indices(self.root, leaf_indices)
        return np.array(leaf_indices)

    def _collect_leaf_indices(self, node, leaf_indices):
        """Helper to recursively collect global indices of leaf nodes."""
        if node.is_leaf():
            leaf_indices.extend(node.global_indices.tolist())
        else:
            for child in node.childrens:
                self._collect_leaf_indices(child, leaf_indices)
    
    def _build_recursive(self, node: BSPNode, N_leaf: int):
        """Recursively build the tree by splitting nodes until leaf conditions are met."""
        if node.depth > self.max_depth:
            self.max_depth = node.depth

        if node.points.shape[0] > N_leaf:
            node.split()
            for child in node.childrens:
                self._build_recursive(child, N_leaf)
                
    def validate_tree_structure(self, N_leaf: int) -> bool:
        """
        Validate that each leaf node has at most N_leaf points.
        
        Parameters:
        - N_leaf: int -> Maximum number of points allowed per leaf node.
        
        Returns:
        - bool -> True if the tree structure is valid, False otherwise.
        """
        def check_node(node):
            if node.is_leaf():
                return len(node.points) <= N_leaf
            return all(check_node(child) for child in node.childrens)
        
        return check_node(self.root)

    def display_indexing(self):
        """Display the indices of points within the leaf nodes."""
        leaf_indices = self.indexing
        plt.scatter(leaf_indices, self.points[leaf_indices, 0], label="$x$")
        plt.scatter(leaf_indices, self.points[leaf_indices, 1], label="$y$")
        plt.legend()
        plt.show()
        
        plt.scatter(range(len(leaf_indices)), leaf_indices)
        plt.show()
    
    def display(self, cmap = cm.hsv, alpha = 0.75):
        """Visualize the BSP tree structure, showing bounding boxes and points."""
        color_map = cmap
        norm = mpl.colors.BoundaryNorm(range(self.max_depth + 1), cmap.N, extend='neither')
        # norm = plt.Normalize(vmin=0, vmax=self.max_depth)
        fig, ax = plt.subplots()
        self.root.display(max_depth=self.max_depth, ax=ax, color_map=cmap, alpha = alpha)
        ax.scatter(*self.points.T, color="red", s=1, zorder=5)
        # f
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])  # Dummy array for the colorbar
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, alpha = alpha)
        cbar.set_label("Depth of Node")
        plt.title(f"BSP Tree - Max Depth: {self.max_depth}")
        plt.show()


# Example usage
if __name__ == '__main__':    
    N_e = 200
    a = 4.0  # semi-major axis
    b = 1.0  # semi-minor axis
    
    N_leaf = 3
    # N_leaf = 10
    N_leaf = 40
    
    # obstacle = Square(N_e=N_e, width=a)
    obstacle = Ellipse(N_e=N_e, a=a, b=b)
    # obstacle = Disc(N_e=N_e, radius=a)
    points = obstacle.y_e
    
    # N = len(points)
    # print(f"{N = }")
    
    M = 10
    N = M//2
    print(list(range(N)))
    print(list(range(N, M)))
    
    # tree = BSPTree(points)
    # tree.build_tree(N_leaf=N_leaf)
    # tree.display(cmap=cmr.lavender, alpha = 0.5)
    # # tree.display_indexing()
    # # print(tree.validate_tree_structure(N_leaf=3))