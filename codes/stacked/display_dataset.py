from matplotlib import pyplot
import sys
import numpy as np
import os
from six.moves import cPickle
from dataset import getDatasets32
from dataset import getDatasets64
from dataset import getDatasets128



# example of loading and plotting the cifar10 dataset
# load the images into memory
trainX = getDatasets32()
# plot images from the training dataset
for i in range(49):
	# define subplot
	pyplot.subplot(7, 7, 1 + i)
	# turn off axis
	pyplot.axis('off')
	# plot raw pixel data
	pyplot.imshow(trainX[100+i])
pyplot.show()
