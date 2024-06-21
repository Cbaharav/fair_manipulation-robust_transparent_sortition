import itertools
import numpy as np

def output_all_vectors(P):
    num_feature_values = [] # number of feature values per feature
    feature_values = []     # feature value names - set to 0,1,2,3... for ease of notation
    for arr in P:
        fv = [i for i in range(len(arr))]
        feature_values.append(fv)
        num_feature_values.append(len(fv))
    feature_values = tuple(feature_values)
    num_feature_values = tuple(num_feature_values)

    # All possible feature vectors one can report
    return [p for p in itertools.product(*feature_values)]

def gini(x, w=None):
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / 
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

