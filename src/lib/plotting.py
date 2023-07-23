import matplotlib.pyplot as plt
import numpy as np

def get_average(history, key):
    """
    Given a historu object and metric key, returns a list
    of the average value for the key at each epoch until
    the maximum epoch trained in the cross-folds
    """
    num_folds = len(history[key])
    vals = []
    epoch = 0
    while True:
        avg_val = 0.
        num_vals = 0
        for i in range(num_folds):
            if len(history[key][i]) > epoch:
                num_vals += 1
                avg_val += history[key][i][epoch]
            # use the last value if we've run out of epochs for this fold
            else:
                avg_val += history[key][i][-1]
        if num_vals == 0:
            break
        else:
            vals.append(avg_val / num_folds)
            epoch += 1
    return np.array(vals)