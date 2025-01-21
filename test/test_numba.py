import numpy as np

from numba import njit


@njit
def do_dot():
    X = np.asfortranarray(np.ones((3, 4)))
    B = np.ones((4, 5))
    X[:, :] @ B[:, :]
    # X @ B


if __name__ == '__main__':
    do_dot()