import numpy as np


# %% Read X and Y
def read_data(source="./data/pathway_activity.csv"):
    xy = np.loadtxt(source, delimiter=',', dtype=np.float32)

    x_train = xy[:913, :-1]
    x_test = xy[913:, :-1]

    # Encode label(0, 1, 2, 3) to one-hot
    label = xy[:, [-1]].flatten().astype(int)
    y = np.zeros((label.size, 4))
    y[np.arange(label.size), label] = 1

    y_train = y[:913, :]
    y_test = y[913:, :]

    return x_train, y_train, x_test, y_test
