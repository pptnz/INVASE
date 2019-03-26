import numpy as np


# %% Read X and Y
def read_data(source="./data/pathway_activity.csv"):
    xy = np.loadtxt(source, delimiter=',', dtype=np.float32)
    np.random.shuffle(xy)

    x_train = xy[:913, :-1]
    y_train = xy[:913, [-1]]

    x_test = xy[913:, :-1]
    y_test = xy[913:, [-1]]

    return x_train, y_train, x_test, y_test
