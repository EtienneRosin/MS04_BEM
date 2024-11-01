# Partially-pivoted Cross Approximation
# ppca

# adaptive cross algorithm


from pprint import pprint
import numpy as np

A = np.zeros((5,5))
A[1, 1] = 1
A[2, 1] = 2
A[3, 2] = 1
A[4, 2] = 1

pprint(A)
print(f"{A.ndim = }")

# def compute_low_rank_approximation(A: np.ndarray):
#     A.ndim 





if __name__ == '__main__':
    # find pivot
    i, j = np.unravel_index(np.argmax(A, axis=None), A.shape)
    # print(f"{ind = }")
    
    print(A[i, :])
    print(A[:, j])
    # print(A[ind[0], :].T @ A[:, ind[1]])
    print(np.dot(A[i, :], A[:, j]))
    print(np.outer(A[:, j], A[i, :]))
    # print(np.dot(A[:, ind[0]], A[ind[1], :]))