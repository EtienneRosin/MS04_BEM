"""
OptiMat: A library for low-rank matrix approximation [1]_.

Features:
- Fully-pivoted Cross Approximation (FPCA)
- Partially-pivoted Cross Approximation (PPCA)
- Adaptive Cross Approximation (ACA)

References
----------
The algorithms implemented are based on:

.. [1] Chaillat, S. "Low-Rank Approximation Methods for Large Matrices." Available at: https://uma.ensta-paris.fr/var/files/chaillat/seance7.pdf
"""

from .utils import generate_matrix_with_rank, argmax_in_subarray