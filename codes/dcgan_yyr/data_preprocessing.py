import os
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

def load_data(data_path):
    """
    载入图片数据
    :return: numpy.array 类型的 data, 形状为 (num, 256, 256, 3), 值域为 0 ~ 255 的整数(uint8类型)
    """
    name_list = os.listdir(data_path)
    data = []
    for name in name_list:
        img = mpimg.imread(os.path.join(data_path, name))
        if img.shape == (256, 256, 3):
            data.append(img)
    return np.array(data)

data_path = '../../data/mountains'
data256 = load_data(data_path)
data128 = []
data64 = []
for i in range(len(data256)):
    img = Image.fromarray(data256[i])
    data128.append(np.array(img.resize([128, 128])))
    data64.append(np.array(img.resize([64, 64])))
data128 = np.array(data128)
data64 = np.array(data64)
np.save(data_path + '128.npy', data128)
np.save(data_path + '64.npy', data64)
