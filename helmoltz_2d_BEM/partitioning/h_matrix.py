import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from helmoltz_2d_BEM.geometry import Disc, Square, Ellipse  
from helmoltz_2d_BEM.partitioning import BSPTree, BSPNode


def node_distance(node_1: BSPNode, node_2: BSPNode) -> float:
    """Compute the minimum distance between the bounding boxes of two BSP nodes.

    Parameters
    ----------
    node_1 : BSPNode
        First node.
    node_2 : BSPNode
        Second node.
    
    Returns
    -------
    float
        Minimum Euclidean distance between the bounding boxes of the two nodes.
    """
    bbox_1 = node_1.bbox
    bbox_2 = node_2.bbox
    squared_distance = np.maximum(0, bbox_1[0, 0] - bbox_2[0, 1])**2 + np.maximum(0, bbox_2[0, 0] - bbox_1[0, 1])**2
    squared_distance += np.maximum(0, bbox_1[1, 0] - bbox_2[1, 1])**2 + np.maximum(0, bbox_2[1, 0] - bbox_1[1, 1])**2
    return np.sqrt(squared_distance)

def nodes_are_eta_admissibles(node_1: BSPNode, node_2: BSPNode, eta: float = 3) -> bool:
    """Return if the nodes are eta-admissible.

    Parameters
    ----------
    node_1 : BSPNode
        _description_
    node_2 : BSPNode
        _description_
    eta : float, optional
        _description_, by default 3

    Returns
    -------
    bool
        _description_
    """
    # print(f"{eta = }")
    return np.minimum(node_1.diameter, node_2.diameter) < eta * node_distance(node_1, node_2)




class HMatrix:
    def __init__(self, bsp_tree: BSPTree) -> None:
        self.bsp_tree = bsp_tree
        self.partitionning = None
        
        # self.sigma = [] # indices of the X
        self.block_indices = []
        self.block_is_admissible = []
        # self.tau = []   # indices of the Y
        pass

    def partition(self, eta = 3):
        """Create the H-matrix partionning
        """
        
        self._recursive_admissible_check(node_1 = self.bsp_tree.root, node_2 = self.bsp_tree.root, eta = eta)
        pass
    
    def _recursive_admissible_check(self, node_1: BSPNode, node_2: BSPNode, eta = 3):
        # print(f"{eta = }")
        if node_1.is_leaf() or node_2.is_leaf():
            self.block_indices.append([np.array(range(node_1.i_min, node_1.i_max)), np.array(range(node_2.i_min, node_2.i_max))])
            self.block_is_admissible.append(False)
        else:
            if nodes_are_eta_admissibles(node_1, node_2, eta = eta):
                self.block_indices.append([np.array(range(node_1.i_min, node_1.i_max)), np.array(range(node_2.i_min, node_2.i_max))])
                self.block_is_admissible.append(True)
            else:
                self._recursive_admissible_check(node_1.childrens[0], node_2.childrens[0], eta)
                self._recursive_admissible_check(node_1.childrens[0], node_2.childrens[1], eta)
                self._recursive_admissible_check(node_1.childrens[1], node_2.childrens[0], eta)
                self._recursive_admissible_check(node_1.childrens[1], node_2.childrens[1], eta)
                
        
    
    # def fill(self, expr: callable) -> None:
    #     """Fill the H-matrix with the values of the expression evaluated at the points of the BSP tree.

    #     Parameters
    #     ----------
    #     expr : callable
    #         _description_
    #     """

    def display_structure(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set(
            xlim = (-1, self.bsp_tree.root.N),
            ylim = (-1, self.bsp_tree.root.N),
            aspect = 'equal')
        # ax.xaxis.set_inverted(True)
        ax.yaxis.set_inverted(True)
        for block_indices, admissible in zip(self.block_indices, self.block_is_admissible):
            block_x_min = block_indices[0].min()
            block_x_max = block_indices[0].max()
            block_y_min = block_indices[1].min()
            block_y_max = block_indices[1].max()
            rect = plt.Rectangle(
                (block_x_min, block_y_min),
                block_x_max - block_x_min,
                block_y_max - block_y_min,
                edgecolor = "#5B9276" if admissible else "#D1453D",
                facecolor = "#5B9276" if admissible else "#D1453D",
                alpha = 0.5
                # linewidth= 5/(self.depth + 1)
            )
            ax.add_patch(rect)
        plt.show()


if __name__ == '__main__':    
    N = 1000
    # lst_x = np.linspace(0, 10, N)
    # lst_y = np.zeros_like(lst_x)
    # points = np.vstack((lst_x, lst_y)).T
    
    a = 4.0  # semi-major axis
    b = 1.0  # semi-minor axis
    obstacle = Ellipse(N_e=N, a=a, b=b)
    # obstacle = Disc(N_e=N, radius=a)
    # obstacle = Square(N_e=N, width=a)
    points = obstacle.y_e
    
    N_leaf = 100
    tree = BSPTree(points)
    tree.build(N_leaf=N_leaf)
    # tree.display()
    
    
    hmat = HMatrix(bsp_tree=tree)
    hmat.partition(eta = 3)
    # print(hmat.block_is_admissible)
    hmat.display_structure()
