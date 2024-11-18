
import numpy as np
from helmoltz_2d_BEM.partitioning import BSPNode, BSPTree
from helmoltz_2d_BEM.geometry import Disc, Square, Ellipse


N = 100
lst_x = np.linspace(0, 10, N)
lst_y = np.zeros_like(lst_x)
points = np.vstack((lst_x, lst_y)).T



N_leaf = 5
tree = BSPTree(points)
tree.build_tree(N_leaf=N_leaf)
    
tree.display()