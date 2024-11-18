import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from helmoltz_2d_BEM.geometry import Disc, Square, Ellipse  
import cmasher as cmr



class BSPNode:
    def __init__(
        self,
        tree,
        depth: int,
        i_min: int,
        i_max: int,
        ) -> None:
        self.tree = tree
        self.depth = depth
        self.i_min = i_min
        self.i_max = i_max
        self.N = i_max - i_min
        self.childrens = []
        self.bbox = self._get_bbox()
        pass
    
    def is_leaf(self):
        """Return True if the node is a leaf (has no children)."""
        return len(self.childrens) == 0
    
    @property
    def diameter(self):
        """Calculate the diameter (diagonal length) of the bounding box."""
        return np.linalg.norm(self.bbox[:, 1] - self.bbox[:, 0])
    
    def _get_bbox(self):
        
        bbox = np.array([
            [self.tree.points[self.tree.sorted_indices[self.i_min:self.i_max], 0].min(), 
             self.tree.points[self.tree.sorted_indices[self.i_min:self.i_max], 0].max()],
            [self.tree.points[self.tree.sorted_indices[self.i_min:self.i_max], 1].min(), 
             self.tree.points[self.tree.sorted_indices[self.i_min:self.i_max], 1].max()]
            ])
        
        # bbox[self.new_splitting_axis, 0] = self.tree.points[self.splitting_axis_sorted_indices][:, self.new_splitting_axis].min()
        # bbox[self.new_splitting_axis, 1] = self.tree.points[self.splitting_axis_sorted_indices][:, self.new_splitting_axis].max()
        # bbox[self.splitting_axis, 0] = self.tree.points[self.splitting_axis_sorted_indices][:, self.splitting_axis].min()
        # bbox[self.splitting_axis, 1] = self.tree.points[self.splitting_axis_sorted_indices][:, self.splitting_axis].max()
        return bbox
    def display(self, max_depth, ax=None, color_map=cmr.lavender, alpha = 0.25):
        """
        Recursively display the node's bounding box and those of its children.

        Parameters:
        - max_depth: int -> Maximum depth of the tree for color scaling.
        - ax: matplotlib.axes.Axes or None -> Axes to plot on (creates new Axes if None).
        - color_map: matplotlib.cm -> Colormap to represent depth.
        """
        display = False
        # print(f"{self.depth = }, {self.bbox = }")
        if ax is None:
            fig, ax = plt.subplots()
            display = True
        
        color = color_map(self.depth / (max_depth ))
        rect = plt.Rectangle(
            (self.bbox[0, 0], self.bbox[1, 0]),
            *np.diff(self.bbox).flatten(),
            edgecolor = color,
            facecolor = color,
            alpha = alpha
            # facecolor="none"
        )
        ax.add_patch(rect)
        
        for child in self.childrens:
            child.display(max_depth=max_depth, ax=ax, color_map=color_map, alpha=alpha)

        ax.set_aspect('equal', 'box')
        if display:
            ax.scatter(*self.tree.points[self.splitting_axis_sorted_indices].T, s = 1)
            plt.show()
            
    def _split(self):
        self.childrens = [
            BSPNode(
                tree=self.tree,
                depth=self.depth+1,
                i_min=self.i_min,
                i_max=self.i_min + self.N//2
            ),
            BSPNode(
                tree=self.tree,
                depth=self.depth+1,
                i_min=self.i_min + self.N//2,
                i_max=self.i_max
            )
        ]
        pass


class BSPTree:
    def __init__(self, points:np.ndarray) -> None:
        self.points = np.array(points, copy = False)
        self.bbox = self._get_bbox()
        splitting_axis = np.argmax(np.diff(self.bbox).flatten()) # 0 if x, 1 if y
       
        self.max_depth = 0
        # Sort the points 2 times to get contiguous points
        # print(points)
        splitting_axis_sorted_indices = np.lexsort((points[:, splitting_axis], points[:, 1 - splitting_axis]))
        # print(f"{splitting_axis_sorted_indices = }")
        self.sorted_indices = np.lexsort(
                (points[splitting_axis_sorted_indices, splitting_axis], 
                 points[splitting_axis_sorted_indices, 1 - splitting_axis])
                )
        # print(points)
        self.root = BSPNode(
            tree=self,
            depth=0,
            i_min=0,
            i_max=len(points)
        )
        pass
    
    def _get_bbox(self):
        bbox = np.array([
            [self.points[:, 0].min(), self.points[:, 0].max()],
            [self.points[:, 1].min(), self.points[:, 1].max()]
            ])
        return bbox
    
    def display(self, cmap = cmr.lavender, alpha = 0.25):
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
    
    def _build_recursive(self, node: BSPNode, N_leaf: int):
        """Recursively build the tree by splitting nodes until leaf conditions are met."""
        if node.depth > self.max_depth:
            self.max_depth = node.depth

        if node.N > N_leaf:
            node._split()
            for child in node.childrens:
                self._build_recursive(child, N_leaf)
    def build(self, N_leaf: int):
        """
        Recursively build the BSP tree until each leaf node has at most N_leaf points.
        
        Parameters:
        - N_leaf: int -> Maximum number of points allowed in each leaf node.
        """
        self._build_recursive(self.root, N_leaf)
        
    def get_leaf_new_indexing(self):
        """Return global indices of all points in leaf nodes as a flattened list."""
        leaf_indices = []
        self._collect_leaf_indices(self.root, leaf_indices)
        return np.array(leaf_indices)
    
    def _collect_leaf_indices(self, node: BSPNode, leaf_indices):
        """Helper to recursively collect global indices of leaf nodes."""
        if node.is_leaf():
            
            leaf_indices.extend(list(range(node.i_min, node.i_max)))
            # leaf_indices.extend(node.new_indexing.tolist())
        else:
            for child in node.childrens:
                self._collect_leaf_indices(child, leaf_indices)


def generate_deformed_disk(N, radius=1, deformation_factor=0.2): 
    angles = np.linspace(0, 2 * np.pi, N, endpoint=True) 
    radii = radius * (1 + deformation_factor * np.sin(3 * angles)) # Déformation du disque 
    x = radii * np.cos(angles) 
    y = radii * np.sin(angles) 
    points = np.column_stack((x, y)) 
    return points

if __name__ == '__main__':
    N = 2000
    # points = generate_deformed_disk(N, deformation_factor=0.25)
    a = 4.0  # semi-major axis
    b = 1.0  # s
    # obstacle = Square(N_e=N, width=a)
    obstacle = Ellipse(N_e=N, a=a, b=b)
    # obstacle = Disc(N_e=N, radius=a)
    points = obstacle.y_e
    
    tree = BSPTree(points)
    print(tree.sorted_indices)
    # tree.root._split()
    tree.build(30)
    tree.display(alpha=0.25)
    
    indexing = tree.get_leaf_new_indexing()
    print(indexing)