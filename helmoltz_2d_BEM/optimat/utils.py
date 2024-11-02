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
    
    Examples
    --------
    >>> m, n, r = 3, 3, 2
    >>> matrix = generate_matrix_with_rank(m, n, r)
    >>> matrix
    [[ 0.49289538 -0.64434182 -0.68066769]
     [-5.27849047 -3.02367366 -1.07246465]
     [ 1.07732423  0.67227925  0.26536084]]
    >>> rank = np.linalg.matrix_rank(matrix)
    >>> rank
    2
    """
    if r > min(m, n):
        raise ValueError("Le rang r doit être inférieur ou égal à min(m, n)")

    # create 2 random matrices of shapes (m, r) and (r, n)
    A = np.random.randn(m, r)
    B = np.random.randn(r, n)
    
    # Matrices product to get a (m,n) matrix of rank r
    matrix = np.dot(A, B)
    
    return matrix

def argmax_in_subarray(a: np._typing.ArrayLike, I: set):
    r"""Find the argmax in a for indices not in I.
    
    This function solves :
    .. math::
        \arg \max_{i \notin I} a_i

    Parameters
    ----------
    a: array_like
        Considered array.
    I: set
        Set of excluded indices.

    Returns
    -------
    i_star: int
        Considered argmax.

    Raises
    ------
    ValueError
        If indices of `I` cover all elements of `a`.

    Notes
    -----
    In case of multiple occurrences of the maximum values, the indices
    corresponding to the first occurrence are returned.
    
    If `I` is empty then the function uses ``np.argmax()``.
    
    We thought to find the argmax by using ``np.argmax()`` on a masked array 
    and then getting the real index within a with ``np.where()``, but as it involves
    a copy of `a`, we thought that it was less optimal than the current 
    implementation.

    Examples
    --------
    >>> I = {1, 2}
    >>> a = np.array([0, 10, 20, 1, 6])
    >>> try:
    ...     i_star = argmax_in_subarray(a=np.abs(a), I=I)
    ...     print(f"{i_star = }, {a[i_star] = }")
    ... except ValueError as e:
    ...     print(e)
    i_star = 4, a[i_star] = 6

    >>> I = {0, 1}
    >>> a = np.array([-5, -10, 15, 20])
    >>> i_star = argmax_in_subarray(a=np.abs(a), I=I)
    >>> print(f"{i_star = }, {a[i_star] = }")
    i_star = 3, a[i_star] = 20
    """
    if len(set(range(a.size)) - I) == 0:
        raise ValueError("Indices of I cover all the a array.")
    if len(I) == 0:
        return np.argmax(a)
    max_value = -np.inf
    i_star = -1

    for i in range(len(a)): 
        if i not in I and a[i] > max_value:
            max_value = a[i]
            i_star = i
    
    return i_star
    


if __name__ == '__main__':
    m, n, r = 3, 3, 2
    matrix = generate_matrix_with_rank(m, n, r)
    print(matrix)

    # # Vérification du rang
    # rank = np.linalg.matrix_rank(matrix)
    # print("Rang de la matrice :", rank)
    
    I = {0, 1}
    I = set()
    a = np.array([-5, -10, 15, 20])
    print(f"{a = }, {a.size = }")

    try:
        i_star = argmax_in_subarray(a=np.abs(a), I=I)
        print(f"{i_star = }, {a[i_star] = }")
    except ValueError as e:
        print(e)
    