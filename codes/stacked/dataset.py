import numpy as np

def getDatasets32():
    data = np.load("./data32_64_128/pics32.npy")
    # data = data/255.0
    return data

def getDatasets64():
    data = np.load("./data32_64_128/pics64.npy")
    # data = data/255.0
    return data

def getDatasets128():
    data = np.load("./data32_64_128/pics128.npy")
    # data = data/255.0
    return data

