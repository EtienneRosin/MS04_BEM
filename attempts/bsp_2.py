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
        depth,
        splitting_axis,
        splitting_axis_sorted_indices,
        i_min,
        i_max,
        parent = None,
        ) -> None:
        self.tree = tree
        self.depth = depth
        self.splitting_axis = splitting_axis
        self.new_splitting_axis = 1 - splitting_axis
        self.splitting_axis_sorted_indices = splitting_axis_sorted_indices
        self.num_points = len(splitting_axis_sorted_indices)
        self.i_min = i_min
        self.i_max = i_max
        
        self.parent = parent
        self.bbox = self._get_bbox()
        self.childrens = [] # we walk through the figure in trigonometric direction
        
    
    def _get_bbox(self):
        bbox = np.zeros((2, 2))
        print(self.splitting_axis_sorted_indices)
        # print(f"{self.splitting_axis_sorted_indices[0] = }")
        # bbox[self.splitting_axis, 0] = self.tree.points[self.splitting_axis_sorted_indices[0], self.splitting_axis]
        # bbox[self.splitting_axis, 1] = self.tree.points[self.splitting_axis_sorted_indices[-1], self.splitting_axis]
        # bbox[self.new_splitting_axis, 0] = self.tree.points[self.splitting_axis_sorted_indices][:, self.new_splitting_axis].min()
        # bbox[self.new_splitting_axis, 1] = self.tree.points[self.splitting_axis_sorted_indices][:, self.new_splitting_axis].max()
        # bbox[self.new_splitting_axis, 0] = self.tree.points[self.splitting_axis_sorted_indices[0], self.splitting_axis]
        # bbox[self.new_splitting_axis, 1] = self.tree.points[self.splitting_axis_sorted_indices[-1], self.splitting_axis]
        # bbox[self.splitting_axis, 0] = self.tree.points[self.splitting_axis_sorted_indices][:, self.new_splitting_axis].min()
        # bbox[self.splitting_axis, 1] = self.tree.points[self.splitting_axis_sorted_indices][:, self.new_splitting_axis].max()
        bbox[self.new_splitting_axis, 0] = self.tree.points[self.splitting_axis_sorted_indices][:, self.new_splitting_axis].min()
        bbox[self.new_splitting_axis, 1] = self.tree.points[self.splitting_axis_sorted_indices][:, self.new_splitting_axis].max()
        bbox[self.splitting_axis, 0] = self.tree.points[self.splitting_axis_sorted_indices][:, self.splitting_axis].min()
        bbox[self.splitting_axis, 1] = self.tree.points[self.splitting_axis_sorted_indices][:, self.splitting_axis].max()
        return bbox
        
    @property
    def diameter(self):
        """Calculate the diameter (diagonal length) of the bounding box."""
        return np.linalg.norm(self.bbox[:, 1] - self.bbox[:, 0])
    
    def is_leaf(self):
        """Return True if the node is a leaf (has no children)."""
        return len(self.childrens) == 0
    
    def display(self, max_depth, ax=None, color_map=cmr.lavender, alpha = 0.25):
        """
        Recursively display the node's bounding box and those of its children.

        Parameters:
        - max_depth: int -> Maximum depth of the tree for color scaling.
        - ax: matplotlib.axes.Axes or None -> Axes to plot on (creates new Axes if None).
        - color_map: matplotlib.cm -> Colormap to represent depth.
        """
        display = False
        print(f"{self.depth = }, {self.bbox = }")
        if ax is None:
            fig, ax = plt.subplots()
            display = True
        
        color = color_map(self.depth / max_depth)
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
        new_splitting_axis_sorted_indices = np.lexsort(
            (self.tree.points[self.splitting_axis_sorted_indices, self.splitting_axis], 
             self.tree.points[self.splitting_axis_sorted_indices, self.new_splitting_axis])
            )
        
        self.childrens = [
            BSPNode(
                tree=self.tree,
                depth=self.depth+1,
                splitting_axis=self.new_splitting_axis,
                splitting_axis_sorted_indices=new_splitting_axis_sorted_indices[:self.num_points//2],
                i_min=self.i_min,
                i_max=self.i_min + self.num_points//2 - 1,
                parent=self
            ),
            BSPNode(
                tree=self.tree,
                depth=self.depth+1,
                splitting_axis=self.new_splitting_axis,
                splitting_axis_sorted_indices=new_splitting_axis_sorted_indices[self.num_points//2:],
                i_min= self.i_max - self.num_points//2,
                i_max=self.i_max,
                parent=self
            )
        ]
        
        
        
        
        
        
        
        
        
        # print(f"{M = }, {0 + M//2 - 1 = }, {M - M//2= }")
        # N//2, N//2 + 1
        ######## sens trigo ::::
        # axe = x : 
        pass
        return new_splitting_axis_sorted_indices


class BSPTree:
    def __init__(self, points: np.ndarray) -> None:
        self.points = np.array(points, copy = False)
        self.bbox = self._get_bbox()
        self.splitting_axis = np.argmax(np.diff(self.bbox).flatten()) # 0 if x, 1 if y
        splitting_axis_sorted_indices = np.lexsort((points[:, self.splitting_axis], points[:, 1 - self.splitting_axis]))
        self.max_depth = 0
        self.root = BSPNode(
            tree = self,
            depth = 0,
            splitting_axis = self.splitting_axis,
            splitting_axis_sorted_indices = splitting_axis_sorted_indices,
            i_min = 0,
            i_max = len(points),
            parent = None
        )
        
    def build_tree(self, N_leaf: int):
        """
        Recursively build the BSP tree until each leaf node has at most N_leaf points.
        
        Parameters:
        - N_leaf: int -> Maximum number of points allowed in each leaf node.
        """
        self._build_recursive(self.root, N_leaf)
        # self.indexing = self.get_leaf_new_indexing()
    
    # def get_leaf_new_indexing(self):
    #     """Return global indices of all points in leaf nodes as a flattened list."""
    #     leaf_indices = []
    #     self._collect_leaf_indices(self.root, leaf_indices)
    #     return np.array(leaf_indices)
    
    # def _collect_leaf_indices(self, node, leaf_indices):
    #     """Helper to recursively collect global indices of leaf nodes."""
    #     if node.is_leaf():
    #         leaf_indices.extend(node.new_indexing.tolist())
    #     else:
    #         for child in node.childrens:
    #             self._collect_leaf_indices(child, leaf_indices)
    
    def _get_bbox(self):
        bbox = np.array([
            [self.points[:, 0].min(), self.points[:, 0].max()],
            [self.points[:, 1].min(), self.points[:, 1].max()]
            ])
        return bbox
    
    def _build_recursive(self, node: BSPNode, N_leaf: int):
        """Recursively build the tree by splitting nodes until leaf conditions are met."""
        if node.depth > self.max_depth:
            self.max_depth = node.depth

        if node.num_points > N_leaf:
            node._split()
            for child in node.childrens:
                self._build_recursive(child, N_leaf)
    
    def display(self, cmap = cmr.lavender, alpha = 0.25):
        """Visualize the BSP tree structure, showing bounding boxes and points."""
        color_map = cmap
        norm = mpl.colors.BoundaryNorm(range(self.max_depth + 2), cmap.N, extend='neither')
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
    
    



def generate_deformed_disk(N, radius=1, deformation_factor=0.2): 
    angles = np.linspace(0, 2 * np.pi, N, endpoint=True) 
    radii = radius * (1 + deformation_factor * np.sin(3 * angles)) # Déformation du disque 
    x = radii * np.cos(angles) 
    y = radii * np.sin(angles) 
    points = np.column_stack((x, y)) 
    return points




if __name__ == '__main__':
    N = 20
    points = generate_deformed_disk(N, deformation_factor=0.25) # Afficher les points pour visualiser la forme
    
    N_e = 200
    a = 4.0  # semi-major axis
    b = 1.0  # s
    # obstacle = Square(N_e=N_e, width=a)
    # obstacle = Ellipse(N_e=N_e, a=a, b=b)
    obstacle = Disc(N_e=N_e, radius=a)
    points = obstacle.y_e
    
    tree = BSPTree(points)
    # print(tree._get_bbox())
    # tree
    
    splitting_axis_sorted_indices = np.lexsort((points[:, tree.splitting_axis], points[:, 1 - tree.splitting_axis]))
    new_splitting_axis_sorted_indices = np.lexsort(
            (points[splitting_axis_sorted_indices, tree.splitting_axis], 
             points[splitting_axis_sorted_indices, 1 - tree.splitting_axis])
            )
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection = '3d')
    # ax.plot(*points[new_splitting_axis_sorted_indices].T, new_splitting_axis_sorted_indices)
    # plt.show()
    
    N = len(points)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.plot(*points[new_splitting_axis_sorted_indices][:N//2].T, new_splitting_axis_sorted_indices[:N//2])
    ax.plot(*points[new_splitting_axis_sorted_indices][N//2-1:].T, new_splitting_axis_sorted_indices[N//2-1:])
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.plot(*points[new_splitting_axis_sorted_indices][:N//4].T, new_splitting_axis_sorted_indices[:N//4])
    ax.plot(*points[new_splitting_axis_sorted_indices][N//4-1:2*(N//4)].T, new_splitting_axis_sorted_indices[N//4-1:2*(N//4)])
    ax.plot(*points[new_splitting_axis_sorted_indices][2*(N//4)-1:3*(N//4)].T, new_splitting_axis_sorted_indices[2*(N//4)-1:3*(N//4)])
    ax.plot(*points[new_splitting_axis_sorted_indices][3*(N//4)-1:].T, new_splitting_axis_sorted_indices[3*(N//4)-1:])
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.plot(*points[new_splitting_axis_sorted_indices[:N//2]].T, new_splitting_axis_sorted_indices[:N//2])
    ax.plot(*points[new_splitting_axis_sorted_indices[N//2-1:]].T, new_splitting_axis_sorted_indices[N//2-1:])
    plt.show()
    
    
    
    
    
    
    
    # lst_x = np.linspace(0, 2, num=N)
    # lst_y = np.linspace(3, 10, num=N)
    # points = np.vstack((lst_x, lst_y)).T
    # print(points)
    
    # tree = BSPTree(points)
    # print(tree._get_bbox())
    # print(tree.splitting_axis)
    # splitting_axis_sorted_indices = np.argsort(points[:, tree.splitting_axis])
    # splitting_axis_sorted_indices = np.lexsort((points[:, tree.splitting_axis], points[:, 1-tree.splitting_axis]))
    # print(points[splitting_axis_sorted_indices])
    
    
    # root = BSPNode(
    #     tree = tree,
    #     depth = 0,
    #     splitting_axis = tree.splitting_axis,
    #     splitting_axis_sorted_indices = splitting_axis_sorted_indices,
    #     i_min = 0,
    #     i_max = len(points),
    #     parent = None
    # )
    
    # M = len(points)
    # print(f"{M = }, {0 + M//2 - 1 = }, {M - M//2= }")
    # print(list(range(M//2)))
    # print(list(range(M//2)))
    # root._split()
    
    # new_splitting_axis_sorted_indices = root.childrens[0]._split()
    # fig = plt.figure()
    # ax = fig.add_subplot(projection = '3d')
    # ax.plot(*points[new_splitting_axis_sorted_indices].T, new_splitting_axis_sorted_indices)
    # plt.show()
    
    # root.display(max_depth=1, alpha=0.25)
    # tree.build_tree(N_leaf=3)
    # tree.display()
    
    
    # plt.show()
    # print(root.bbox)
    # points = generate_deformed_square(N, deformation_factor=0.1) # Afficher les points pour visualiser la forme 
    # points = generate_star(N)
    # plt.figure(figsize=(6, 6)) 
    # plt.plot(points[:, 0], points[:, 1], 'o-', markersize=2) 
    # plt.gca().set_aspect('equal', adjustable='box') 
    # plt.title('Disque Déformé') 
    # plt.show() # Afficher les premiers points générés 
    # print(points[:10])
   