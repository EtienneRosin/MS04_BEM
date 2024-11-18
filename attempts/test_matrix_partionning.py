import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from helmoltz_2d_BEM.geometry import Disc, Square, Ellipse  
from helmoltz_2d_BEM.partitioning import BSPTree, BSPNode


def node_distance(node_1: BSPNode, node_2: BSPNode):
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

def nodes_are_eta_admissibles(node_1: BSPNode, node_2: BSPNode, eta: float = 3):
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
    _type_
        _description_
    """
    return np.minimum(node_1.diameter, node_2.diameter) < eta * node_distance(node_1, node_2)



if __name__ == '__main__':
    p1 = np.array([1, 1])
    p2 = np.array([2, 2])
    p3 = np.array([4, 2 + 2])
    p4 = np.array([6, 3 + 2])
    
    node_1 = BSPNode(points = np.array([p1, p2]), indices_x_sorted=[0, 1], indices_y_sorted=[0, 1], global_indices=[1, 2])
    node_2 = BSPNode(points = np.array([p3, p4]), indices_x_sorted=[0, 1], indices_y_sorted=[0, 1], global_indices=[3, 4])
    
    print(node_distance(node_1, node_2))
    # print(f"{2 * np.sqrt(2) = }")
    print(nodes_are_eta_admissibles(node_1, node_2))