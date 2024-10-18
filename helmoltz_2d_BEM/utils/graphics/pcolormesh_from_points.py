import numpy as np
import matplotlib.pyplot as plt



def prepare_points_for_pcolormesh(lst_x: np.ndarray, lst_y: np.ndarray, lst_z: np.ndarray):
    if lst_x.shape != lst_y.shape or lst_x.shape != lst_z.shape:
        raise ValueError("Shapes of lst_x, lst_y and lst_z should be the same")
    # X, Y = np.meshgrid(lst_x, lst_y, indexing='ij')
    x_unique = np.unique(lst_x)
    y_unique = np.unique(lst_y)
    Z = np.zeros((len(y_unique), len(x_unique)), dtype = lst_z.dtype)
    
    for i, y in enumerate(y_unique):
        for j, x in enumerate(x_unique):
            # Trouver les indices dans self.domain.nodes qui correspondent à (x, y)
            indices = np.where((lst_x == x) & (lst_y == y))
            if len(indices[0]) > 0:
                Z[i, j] = lst_z[indices[0][0]]
    
    return x_unique, y_unique, Z


def pcolormesh_from_points(lst_x: np.ndarray, lst_y: np.ndarray, lst_z: np.ndarray, **pcolormesh_kwargs):
    if lst_x.shape != lst_y.shape or lst_x.shape != lst_z.shape:
        raise ValueError("Shapes of lst_x, lst_y and lst_z should be the same")