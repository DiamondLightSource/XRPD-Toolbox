import numpy as np


def normalise_to(data: np.ndarray, minval: int | float = 0) -> np.ndarray:
    """
    normalises an array
    minval is  the minimum value that the
    processed array is scaled to.
    """

    return (data - minval) / (np.max(data) - minval)
