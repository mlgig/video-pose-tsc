from itertools import combinations
import numpy as np


def get_combinations(arr, r):
    r_combinations = list(combinations(arr, r))
    r_combinations = [i[0] + "vs" + i[1] for i in r_combinations]
    r_combinations.sort()
    return r_combinations


def nested_to_3d_numpy(X, a=None, b=None):
    """Convert pandas DataFrame (with time series as pandas Series in cells) into NumPy ndarray with shape (n_instances, n_columns, n_timepoints).

    Parameters
    ----------
    X : pandas DataFrame, input
    a : int, first row (optional, default None)
    b : int, last row (optional, default None)

    Returns
    -------
    NumPy ndarray, converted NumPy ndarray
    """
    return np.stack(
        X.iloc[a:b].applymap(lambda cell: cell.to_numpy()).apply(lambda row: np.stack(row), axis=1).to_numpy())


def get_segments(weights, threshold):
    """
    Function to get the indices of segments having values greater than the threshold
    """
    marker_list = [True if i >= threshold else False for i in weights]
    i = 0
    final_pairs = []
    while i < len(weights):
        if marker_list[i]:
            start = i
            while i < len(weights) and marker_list[i]:
                i = i + 1
            end = i - 1
            if end-start > 1:
                final_pairs.append(start)
                final_pairs.append(end)
        i = i + 1
    return np.array(final_pairs)


def get_top_values(weights, top_k=4):
    """
    Function to get the topk values and their indices from an array
    """
    top_idx = np.argsort(weights)[-top_k:]
    top_idx = np.flip(top_idx)
    top_values = [weights[i] for i in top_idx]
    return top_idx, top_values


if __name__ == "__main__":
    weights = np.array([10, 20, 80, 81, 89, 1, 2, 90, 91, 100, 1, 2, 80])
    print(get_top_values(weights))


