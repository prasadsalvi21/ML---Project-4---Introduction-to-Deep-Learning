
# coding: utf-8

# In[1]:

import tensorflow as tf
import cv2
import os, pickle, time, gzip, numpy as np
from PIL import Image
import math
#from tflearn.data_augmentation import ImageAugmentation


# In[2]:

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# In[3]:

os.chdir("/Users/sarahmullin/Desktop/CSE_574/Proj_3")


# In[4]:

def reformat_tf(dataset, labels):
    # dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


# In[5]:

def resize_and_scale(img, size):
    img = cv2.resize(img, size)
    return np.array(img, "float32")


# In[6]:

def load_data(attr_name, sz):
    print("Loading Data.................")
    data = []
    label = []
    attribute = []
    path_to_data = "./img_align_celeba/"
    img_list = os.listdir(path_to_data)
 #   sz = (27, 33)
    for name in sorted(img_list):
        if '.jpg' in name:
            img = cv2.imread(path_to_data + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = resize_and_scale(img, sz)
            data.append(resized_img.flatten())
            #data.append(img.flatten())
    data = np.array(data)
    print("Loading Label.................")
    f = open('./list_attr_celeba.txt', 'r')
    text = f.readlines()
    text = np.array(text)
    attr2idx = dict()
    for i, attr in enumerate(text[1].split()):
        attr2idx[attr] = i
    attr_index = attr2idx[attr_name]#'Eyeglasses'
    for i in text[2:]:
        value = i.split()
        attribute.append(value[attr_index + 1]) #First index is image name
    attribute = np.array(attribute,dtype= np.float32)
    print("Converting Label.................")
    for i in range(0,len(attribute)):
        if (attribute[i] == 1):
            label.append([1, 0])
        else:
            label.append([0, 1])
    label = np.array(label)
    #usps_dataset, usps_label = reformat_tf(usps_data, usps_label)
    return data, label


# In[8]:

#Generate random weights based on shape
def generateWeigths(shape):
    weight = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(weight)

#Generate bias weights based on shape
def generateBias(shape):
    bias = tf.constant(0.1, shape=shape)
    return tf.Variable(bias)

#Apply convolution to image x with filter W and move the strides 1 pixel
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#Max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# In[9]:

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


# In[14]:

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# In[ ]:

def grayscale():
    sz=(27,33)
    image, label = load_data('Eyeglasses', sz)
    train_limit = int(math.floor(0.8 * len(image)))
    validate_limit = int(math.floor(0.1*len(image)))

    train_set_img = image[0:train_limit,:]
    validate_set_img = image[(train_limit+1):(train_limit+validate_limit),:]
    test_set_img = image[(train_limit+validate_limit+1):,:]
    
    train_set_label = label[0:train_limit]
    validate_set_label = label[(train_limit+1):(train_limit+validate_limit)]
    test_set_label = label[(train_limit+validate_limit+1):]
    
    #take a subset of training dataset to find parameters
    train_limit_sm = int(math.floor(0.8 * len(image))*0.5)
    #print (train_limit_sm)
    train_set_img_sm = image[0:train_limit_sm,:]
    train_set_label_sm = label[0:train_limit_sm]
    
    # Tuple with height and width of images used to reshape arrays.
    img_shape = sz
    #img_shape = (89, 109)
    # Images are stored in one-dimensional arrays of this length.
    img_size_flat = img_shape[0] * img_shape[1]
    # Number of colour channels for the images: 1 channel for gray-scale.
    num_channels = 1
    # Number of classes, glasses or no glasses.
    num_classes = 2

    # Convolutional Layer 1.
    filter_size1 = 3          # Convolution filters are 5 x 5 pixels.
    num_filters1 = 32         # There are 32 of these filters.
    #num_filters1 = 16
    # Convolutional Layer 2.
    filter_size2 = 3          # Convolution filters are 5 x 5 pixels.
    num_filters2 = 64         # There are 64 of these filters.
    #num_filters2 = 32
    #Convolutional Layer 3.
    #filter_size3=3
    #num_filters3=64

    # Fully-connected layer.
    fc_size = 1024             # Number of neurons in fully-connected layer.
    #fc_size = 128
    #Learning rate
    learning_rate = 1e-4
    epochs = 20000
    display = 1000
    
    #Placeholders to hold images and labels
    x = tf.placeholder(tf.float32, [None, img_size_flat])

    y_ =  tf.placeholder(tf.float32, [None, num_classes])
    
    #Convolution Layer 1, finds 32 features for each 5x5
    #Weights: patch_size, patch_size, i/p_channel, o/p_channel
    w_conv1 = generateWeigths([filter_size1, filter_size1, num_channels, num_filters1])
    b_conv1 = generateBias([num_filters1])

    #Reshape to shape any, width, height, colour_channel
    x_image = tf.reshape(x, [-1, img_shape[0], img_shape[1], num_channels])

    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    #Convolution Layer 2, finds 64 features for each 5x5
    #Weights: patch_size, patch_size, i/p_channel, o/p_channel
    # 32 channels: 32*64 filters: Take 1,1 2,1.. 32,1 and apply to 1st pixel of each channel and sum to calculate on pixel. Repeat for depth 64
    w_conv2 =  generateWeigths([filter_size2, filter_size2, num_filters1, num_filters2])
    b_conv2 = generateBias([num_filters2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #Convolution Layer 3,
    #w_conv3 =  generateWeigths([filter_size3, filter_size3, num_filters2, num_filters3])
    #b_conv3 = generateBias([num_filters3])

    #h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)
    
    
    layer_flat, num_features = flatten_layer(h_pool2)

    W_fc1 = generateWeigths([num_features, fc_size])
    b_fc1 = generateBias([fc_size])

    h_fc1 = tf.nn.relu(tf.matmul(layer_flat, W_fc1) + b_fc1)

    #Drop out
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = generateWeigths([fc_size, num_classes])
    b_fc2 = generateBias([num_classes])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    ##cross_entropy could be sigmoid function since now only one output
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            batch = next_batch(50, train_set_img, train_set_label);
            if epoch % display == 0:
                cost = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(cost))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        print('Validation accuracy %g' % accuracy.eval(feed_dict={
            x: validate_set_img, y_: validate_set_label, keep_prob: 1.0}))
        print('Test accuracy %g' % accuracy.eval(feed_dict={
            x: test_set_img, y_: test_set_label, keep_prob: 1.0}))




