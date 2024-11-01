"""
ppac.py module.
Partially-pivoted Cross Approximation (PPCA) algorithm.
"""
import numpy as np
from helmoltz_2d_BEM.optimat import generate_matrix_with_rank

def PPCA(A: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Compute the Partially-pivoted Cross Approximation (PPCA) of :math:`\mathbb{A}` (supposed to be of rank :math:`r`).
    This method finds an approximation A_k of A with a low rank by iteratively 
    subtracting outer products based on fully pivoted elements.
    
    The skeleton decomposition of :math:`\mathbb{A}` is given by:
    .. math:: \mathbb{A} \approx \mathbb{C} \hat{\mathbb{A}}^{-1}\mathbb{R}
    Where:
    - :math:`\mathbb{C}` are the columns of the decomposition
    - :math:`\mathbb{R}` are the rows of the decomposition
    - :math:`\hat{\mathbb{A}}` is the invertible :math:`r \times r` submatrix of the decomposition.
    
    Parameters
    ----------
    A : np.ndarray, shape (m, n)
        Input matrix for low-rank approximation.
    tol : float, optional
        Tolerance threshold for approximation. The algorithm stops when 
        the largest pivot in R_k is below this threshold. Default is 1e-10.
    
    Returns
    -------
    A_hat : np.ndarray
        The invertible :math:`r \times r` submatrix of the decomposition.
    C : np.ndarray
        Columns of the decomposition.
    R : np.ndarray
        Rows of the decomposition.
    
    Raises
    ------
    ValueError
        If A is not a 2-dimensional matrix.
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"A should be a matrix of ndim = 2 (here {A.ndim = })")

    # Initialization
    A_k = np.zeros_like(A, dtype=float)
    R_k = np.copy(A)
    I = set()
    J = set()
    i = 0  # initialize at first row
    row_mask = np.full(A.shape[0], fill_value=True)
    row_mask[i] = False
    
    sigma = set(np.arange(A.shape[0])) # all rows indices
    tau = set(np.arange(A.shape[1])) # all columns indices
    
    while True:
        # Add current row index to I
        I.add(i)
        pivot_row = R_k[i, :]
        # Partial pivot: find column index with largest absolute value in row i
        j = np.argmax(np.abs(pivot_row))
        pivot = pivot_row[j]
        pivot_col = R_k[:, j]
        
        if np.abs(pivot) < tol:
            rows_set_difference = sigma - I
            if len(rows_set_difference) == 0: # meaning its empty so check convergence
                ind = np.unravel_index(np.argmax(np.abs(R_k), axis=None), R_k.shape)    
                if np.abs(R_k[ind]) < tol:
                    break
                else:
                    raise ValueError("No convergence")
                
            i = min(rows_set_difference)
            
        else:
            # Add column index to J
            J.add(j)
            
            col_row_product = np.outer(pivot_col, pivot_row)/pivot
            
            
            R_k -= col_row_product
            A_k += col_row_product
            
            print(f"{I = }, {J = }")
            i = np.argmax(np.abs(pivot_col))
            print(R_k, "\n---------------------")
        
        if np.abs(R_k[i,j]) < tol:
            break           
        
    
    # # Sort indices
    I = sorted(I)
    J = sorted(J)
    # print(f"{I = }, {J = }")

    # Extract the matrices for the decomposition
    A_hat = A[np.ix_(I, J)]
    C = A[:, J]
    R = A[I, :]
    # print(R_k)

    return A_hat, C, R

# Test cases (for debugging or verification)
if __name__ == '__main__':
    A = np.array([
        [6.5, 31, -14, -43],
        [9.1, -3, 11, 31],
        [17.6, -16, 28, 80],
        [26.2, 50, -7, -26],
    ])
    
    # A = np.array([
    #     [0., 0., 0., 0., 0.],
    #     [0., 1., 0., 0., 0.],
    #     [0., 2., 0., 0., 0.],
    #     [0., 0., 1., 0., 0.],
    #     [0., 0., 1., 0., 0.]])
    # I = np.array([1, 3])  # Indices des lignes à exclure
    # A_without_I = np.delete(A, I, axis=0)  # Supprime les lignes spécifiées dans I

    # print("Matrice sans les lignes d'indice I :\n", A_without_I)
    # row_mask = np.full(A.shape[0], fill_value=True)
    # row_mask[0] = False
    # row_mask[]
    # print(A[row_mask, :])
    # #
    
    
    print("Original matrix:\n", A)
    print("Of rank:", np.linalg.matrix_rank(A))
    A_hat, C, R = PPCA(A)
    print("A_hat (submatrix):\n", A_hat)
    print("of rank :", np.linalg.matrix_rank(A_hat))
    print("C (columns):\n", C)
    print("R (rows):\n", R)

    # # Validate approximation by reconstructing A
    # reconstructed_A = C @ np.linalg.inv(A_hat) @ R
    # print("Approximated matrix:\n", reconstructed_A)
    # print("Error norm:", np.linalg.norm(A - reconstructed_A))