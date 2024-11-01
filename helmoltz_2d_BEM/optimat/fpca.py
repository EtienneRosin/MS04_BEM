"""
fcpa.py module.
Implementation of the Fully-pivoted Cross Approximation (FPCA) algorithm.
"""
import numpy as np
from helmoltz_2d_BEM.optimat import generate_matrix_with_rank

def FPCA(A: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Compute the Fully-pivoted Cross Approximation (FPCA) of :math:`\mathbb{A}` (supposed to be of rank :math:`r`).
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
    I = []
    J = []

    while True:
        # Find pivot
        i, j = np.unravel_index(np.argmax(np.abs(R_k), axis=None), R_k.shape)
        pivot_value = R_k[i, j]

        # Stop if the pivot is below the tolerance level
        if np.abs(pivot_value) < tol:
            break
        I.append(i)
        J.append(j)

        gamma = 1 / pivot_value
        pivot_col = R_k[:, j]
        pivot_row = R_k[i, :]

        # Compute the outer product and update matrices
        col_row_product = gamma * np.outer(pivot_col, pivot_row)
        R_k -= col_row_product
        A_k += col_row_product

    # Sort indices
    I = sorted(I)
    J = sorted(J)

    # Extract the matrices for the decomposition
    A_hat = A[np.ix_(I, J)]
    C = A[:, J]
    R = A[I, :]

    return A_hat, C, R

# Test cases (for debugging or verification)
if __name__ == '__main__':
    A = np.array([
        [6.5, 31, -14, -43],
        [9.1, -3, 11, 31],
        [17.6, -16, 28, 80],
        [26.2, 50, -7, -26],
    ])
    
    print("Original matrix:\n", A)
    print("Of rank:", np.linalg.matrix_rank(A))
    A_hat, C, R = FPCA(A)
    print("A_hat (submatrix):\n", A_hat)
    print("of rank :", np.linalg.matrix_rank(A_hat))
    print("C (columns):\n", C)
    print("R (rows):\n", R)

    # Validate approximation by reconstructing A
    reconstructed_A = C @ np.linalg.inv(A_hat) @ R
    print("Approximated matrix:\n", reconstructed_A)
    print("Error norm:", np.linalg.norm(A - reconstructed_A))