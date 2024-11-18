import numpy as np
from helmoltz_2d_BEM.partitioning import BSPNode, BSPTree
from helmoltz_2d_BEM.geometry import Disc, Square, Ellipse



N = 100
lst_x = np.linspace(0, 10, N)
lst_y = np.zeros_like(lst_x)
points = np.vstack((lst_x, lst_y)).T



a = 4.0  # semi-major axis
b = 1.0  # semi-minor axis
obstacle = Ellipse(N_e=N, a=a, b=b)
# obstacle = Square(N_e=N_e, width=a)
# obstacle = Disc(N_e=N_e, radius=a)
points = obstacle.y_e



N_leaf = 5
if __name__ == '__main__':
    tree = BSPTree(points)
    tree.build_tree(N_leaf=N_leaf)
    
    tree.display()