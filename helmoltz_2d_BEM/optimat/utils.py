"""
Modules defining some utility functions of the optimat librairy.
"""
import numpy as np

def generate_matrix_with_rank(m, n, r):
    r"""Generate a matrix of shape (m,n) with a rank exactly equal to r.
    
    Parameters
    ----------
    m : int
        Number of rows of the matrix.
    n : int
        Number of columns of the matrix.
    r : int
        The desired matrix rank (should be lesser of equal to min(m, n)).
    
    Returns
    -------
    ``(m,n)`` matrix of rank ``r``.
    
    Raises
    ------
    ValueError
        If r > min(m,n).
    """
    if r > min(m, n):
        raise ValueError("Le rang r doit être inférieur ou égal à min(m, n)")

    # create 2 random matrices of shapes (m, r) and (r, n)
    A = np.random.randn(m, r)
    B = np.random.randn(r, n)
    
    # Matrices product to get a (m,n) matrix of rank r
    matrix = np.dot(A, B)
    
    return matrix


if __name__ == '__main__':
    m, n, r = 10, 10, 5
    matrix = generate_matrix_with_rank(m, n, r)
    print("Matrice générée de rang 5 :\n", matrix)

    # Vérification du rang
    rank = np.linalg.matrix_rank(matrix)
    print("Rang de la matrice :", rank)