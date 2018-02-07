import numpy as np


def random_split_data(data, label, proportion):
    """
    Split two numpy arrays into two parts of `proportion` and `1 - proportion`

    Args:
        - data: numpy array, to be split along the first axis
        - proportion: a float less than 1
    """
    assert data.shape[0] == label.shape[0]
    size = data.shape[0]
    np.random.seed(666)
    s = np.random.permutation(size)
    split_idx = int(proportion * size)
    return data[s[:split_idx]], label[s[:split_idx]], data[s[split_idx:]], label[s[split_idx:]]