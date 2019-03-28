import numpy as np


# %% Read X and Y
def read_data(source="./data/pathway_activity.csv"):
    xy = np.loadtxt(source, delimiter=',', dtype=np.float32)

    boundary = len(xy) * 9 // 10

    x_train = xy[:boundary, :-1]
    x_test = xy[boundary:, :-1]

    # Encode label(0, 1, 2, 3) to one-hot
    label = xy[:, [-1]].flatten().astype(int)
    y = np.zeros((label.size, 4))
    y[np.arange(label.size), label] = 1

    y_train = y[:boundary, :]
    y_test = y[boundary:, :]

    return x_train, y_train, x_test, y_test
