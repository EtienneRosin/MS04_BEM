import numpy as np
from scipy.special import roots_legendre, eval_legendre


# quadrature_points = np.sqrt(1/3, dtype = float) * np.array([-1, 1], dtype=float)
# quadrature_weights = np.array([1, 1])

# if __name__ == '__main__':
#     print(f"{quadrature_weights = }")
#     print(f"{quadrature_points = }")

quadrature_points, quadrature_weights = roots_legendre(2)

if __name__ == '__main__':
    print(f"{quadrature_weights = }")
    print(f"{quadrature_points = }")