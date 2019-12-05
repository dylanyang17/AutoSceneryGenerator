import numpy as np

def getDatasets():
    data = np.load("./data128/pics.npy")
    # data = data/255.0
    return data