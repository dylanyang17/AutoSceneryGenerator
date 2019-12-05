# from 300*300 jpg to numpy array
from PIL import Image
import os
import numpy as np


dirname = './proc_out128'
outDir = './data128'
cnt = 0
allData = []
for filename in os.listdir(dirname):
    filePath = os.path.join(dirname, filename)
    with Image.open(filePath) as img:
        img = np.array(img)
        if img.shape == (128, 128, 3):
            allData.append(img)
            cnt += 1
        else:
            print(img.shape)

print(len(allData))
allData = np.stack(allData)
outFilename = os.path.join(outDir, 'pics.npy')
np.save(outFilename, allData)
