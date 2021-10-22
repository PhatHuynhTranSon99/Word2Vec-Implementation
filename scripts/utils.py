import pickle
import numpy as np

def save_pickle(item, path):
    with open(path, "wb") as handle:
        pickle.dump(item, handle, pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    with open(path, "rb") as handle:
        item = pickle.load(handle)

    return item

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

