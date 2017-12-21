import tensorflow as tf
import cv2
import os, pickle, time, gzip, numpy as np
from PIL import Image
import math
import Gray
import Color
from __future__ import division, print_function, absolute_import

from skimage import color, io
#from scipy.misc import imresize
import skimage

from sklearn.cross_validation import train_test_split
from glob import glob

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy
#from tflearn.data_augmentation import ImageAugmentation


if __name__ == "__main__":
    print("********************************************************************************")
    print("UBitName:prasadde")
    print("personNumber:50207353")
    print("UBitName:veerappa")
    print("personNumber:50247314")
    print("UBitName:sarahmul")
    print("personNumber:34508498")
    print("********************************************************************************")
	print("Gray Scale Model")
	Gray.grayscale()
	print("********************************************************************************")
	print("Color Model")
	Color.color()