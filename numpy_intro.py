# numpy_intro.py
"""Python Essentials: Intro to NumPy.
<Name>
<Class>
<Date>
"""
import numpy as np

def prob1():
    """Define the matrices A and B as arrays. Return the matrix product AB."""
    A = np.array([[3, -1, 4], [1, 5, -9]])
    B = np.array([[2, 6, -5, 3], [5, -8, 9, 7], [9, -3, -2, -3]])
    return np.dot(A, B)
    raise NotImplementedError("Problem 1 Incomplete")


def prob2():
    """Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A."""
    A = np.array([[3, 1, 4], [1, 5, 9], [-5, 3, 1]])
    aCubed = np.dot(A, A)
    aCubed = np.dot(aCubed, A)
    aCubed = -1 * aCubed
    aSquared = np.dot(A, A)
    aSquared = 9 * aSquared
    aModified = -15 * A
    return aCubed + aSquared + aModified
    raise NotImplementedError("Problem 2 Incomplete")


def prob3():
    """Define the matrices A and B as arrays. Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    A = np.ones((7, 7), dtype = np.int64)
    A = np.triu(A)
    C = np.ones((5, 5), dtype = np.int64)
    C = np.triu(C)
    cDiag = np.diag([5]*7)
    C = C - cDiag
    D = np.ones((-1, -1), dtype = np.int64)
    D = np.tril(D)
    B = D + C
    product = np.dot(A, B)
    product = np.dot(product, A)
    product = product.astype(np.int64)
    return product
    raise NotImplementedError("Problem 3 Incomplete")


def prob4(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    aModified = A
    mask = A < 0
    aModified[mask] = 0
    return aModified
    raise NotImplementedError("Problem 4 Incomplete")


def prob5():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    A = np.array([[0, 2, 4], [1, 3, 5]])
    B = np.ones((3, 3), dtype = np.int)
    B = np.tril(B)
    C = np.diag([-2] * 3)
    one = np.vstack((np.zeros(3), A, B))
    two = np.vstack((np.column_stack(np.transpose(A), np.zeros(3)), np.zeros((5, 3), dtype = np.int)))
    three = np.vstack((np.diag([1]*3), np.zeros((2, 3), dtype = np.int), C))
    return np.hstack((one, two, three))
    raise NotImplementedError("Problem 5 Incomplete")


def prob6(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    matrixA = A
    sum = matrixA.sum(axis = 1)
    sum = sum.astype(np.float)
    sum = sum.reshape(-1, 1)
    return A / sum
    raise NotImplementedError("Problem 6 Incomplete")


def prob7():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """
    raise NotImplementedError("Problem 7 Incomplete")
