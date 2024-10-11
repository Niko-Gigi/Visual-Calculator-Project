import numpy as np

def transform_2dvector(matrix, vector):
    """
    Transform a vector using a 2x2 matrix.
    
    Args:
        matrix (np.array): A 2x2 transformation matrix
        vector (np.array): A 2D vector
    
    Returns:
        np.array: The transformed vector
    """
    return np.dot(matrix, vector)