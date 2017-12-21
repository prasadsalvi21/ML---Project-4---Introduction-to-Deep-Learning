
# coding: utf-8

# In[1]:

from __future__ import division, print_function, absolute_import

from skimage import color, io
#from scipy.misc import imresize
import skimage
import numpy as np
from sklearn.cross_validation import train_test_split
import os
from glob import glob

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy
import math


# In[2]:

#os.chdir("/Users/sarahmullin/Desktop/CSE_574/Proj_3")


# In[3]:

def color():

    ###################################
    ### Import picture files 
    ###################################

    files_path = "./img_align_celeba/"

    glasses_files_path = os.path.join(files_path, '*.jpg')

    glasses_files = sorted(glob(glasses_files_path))

    n_files = len(glasses_files)
    #print(n_files)
    #size_image1 =178
    #size_image2=218
    size_image1=27
    size_image2=33
    allX = np.zeros((n_files, size_image1, size_image2, 3), dtype='float32')
    ally = np.zeros(n_files)
    count = 0
    for f in glasses_files:
        try:
            img = io.imread(f)
            new_img = skimage.transform.resize(img, (27, 33, 3))
            allX[count] = np.array(new_img)
            ally[count] = 0
            count += 1
        except:
            continue
    attribute=[]
    g = open('./list_attr_celeba.txt', 'r')
    text = g.readlines()
    text = np.array(text)
    attr2idx = dict()
    for i, attr in enumerate(text[1].split()):
        attr2idx[attr] = i
    attr_index = attr2idx['Eyeglasses']#'Eyeglasses'
    for i in text[2:]:
        value = i.split()
        attribute.append(value[attr_index + 1]) #First index is image name
    attribute = np.array(attribute,dtype= np.float32)
    #print("Converting Label.................")
    for i in range(0,len(attribute)):
        if (attribute[i] == 1):
            ally[i]=1
        else:
            ally[i]=0
    ally = np.array(ally)
    
    ########break up data into training, validation, and test sets
    train_limit = int(math.floor(0.8 * len(allX)))
    validate_limit = int(math.floor(0.1*len(allX)))
    #print (train_limit, validate_limit)


    X = allX[0:train_limit,:,]
    X_validation = allX[(train_limit+1):(train_limit+validate_limit),:,]
    X_test = allX[(train_limit+validate_limit+1):,:,]
    
    Y = ally[0:train_limit]
    Y_validation = ally[(train_limit+1):(train_limit+validate_limit)]
    Y_test = ally[(train_limit+validate_limit+1):]

    # encode the Ys
    Y = to_categorical(Y, 2)
    Y_test = to_categorical(Y_test, 2)
    Y_validation = to_categorical(Y_validation, 2)

    #take a subset of training dataset to find parameters
    x_sm = int(math.floor(0.8 * len(allX))*0.5)
    print (x_sm)
    X_sm = allX[0:x_sm,:,]
    Y_sm = ally[0:x_sm]
    Y_sm=to_categorical(Y_sm, 2)


    print (X.shape, Y.shape, allX.shape, ally.shape, X_sm.shape)
    
    ###################################
    # Image transformations
    ###################################

    # normalisation of images
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Create extra synthetic training data by flipping & rotating images
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)
    
    ###################################
    # Define network architecture
    ###################################

    # Input is a 27x33 image with 3 color channels (red, green and blue)
    network = input_data(shape=[None, 27, 33, 3])
                     #,data_preprocessing=img_prep,
                     #data_augmentation=img_aug)

    # 1: Convolution layer with 32 filters, each 3x3x3
    conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')

    # 2: Max pooling layer
    network = max_pool_2d(conv_1, 2)

    # 3: Convolution layer with 64 filters
    conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')

    #4: Convolution layer with 64 filters
    conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3')

    # 5: Max pooling layer
    network = max_pool_2d(conv_3, 2)

    # 6: Fully-connected 512 node layer
    network = fully_connected(network, 1024, activation='relu')

    # 7: Dropout layer to combat overfitting
    network = dropout(network, 0.5)

    # 8: Fully-connected layer with two outputs
    network = fully_connected(network, 2, activation='softmax')

    # Configure how the network will be trained
    acc = Accuracy(name="Accuracy")
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001, metric=acc)

    # Wrap the network in a model object
    model = tflearn.DNN(network, checkpoint_path='model_glasses_6.tflearn', max_checkpoints = 3,
                        tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')
    ###################################
    # Train model for 1000 epochs
    ###################################
    model.fit(X_sm, Y_sm, validation_set=(X_validation, Y_validation), batch_size=50,
          n_epoch=1000, run_id='model_glasses_6', show_metric=True)

    model.save('model_glasses_6_final.tflearn')
    
    # Evaluate model
    score = model.evaluate(X_test, Y_test)
    print('Test accuarcy: %0.4f%%' % (score[0] * 100))

    
    


# In[ ]:

color()

