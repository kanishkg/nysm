
import tensorflow as tf
from utils import *

def temp_encoder(inputs):
    c3d_npy_path = 'c3d.npy'
    if c3d_npy_path is not None:
        data_dict = np.load(c3d_npy_path, encoding='latin1').item()
    with tf.device('/gpu:0'):
        with tf.variable_scope('encoder3d_1'):
            conv1_1 = conv3d_layer2(data_dict,inputs, 3, 64 ,"conv1_1")
            pool1 = max_pool3d1(conv1_1, 'pool1')

        with tf.variable_scope('encoder3d_2'):
    	    conv2_1 = conv3d_layer2(data_dict,pool1, 64, 128, "conv2_1")
            norm = tf.contrib.slim.batch_norm(conv2_1)
            pool2 = max_pool3d(norm, 'pool2')

        with tf.variable_scope('encoder3d_3'):
            conv3_1 = conv3d_layer2(data_dict,pool2, 128, 256, "conv3_1")
            conv3_2 = conv3d_layer2(data_dict,conv3_1, 256, 256, "conv3_2")
            norm = tf.contrib.slim.batch_norm(conv3_2)
            pool3 = max_pool3d(norm, 'pool3')
    with tf.device('/gpu:1'):
        with tf.variable_scope('encoder3d_4'):
            conv4_1 = conv3d_layer2(data_dict,pool3, 256, 512, "conv4_1")
            conv4_2 = conv3d_layer2(data_dict,conv4_1, 512, 512, "conv4_2")
            norm = tf.contrib.slim.batch_norm(conv4_2)
            pool4 = max_pool3d(norm,'pool4')

        with tf.variable_scope('encoder3d_5'):
            conv5_1 = conv3d_layer2(data_dict,pool4, 512, 512, "conv5_1")
            conv5_2 = conv3d_layer2(data_dict,conv5_1, 512, 512, "conv5_2")
            norm = tf.contrib.slim.batch_norm(conv5_2)
            pool5 = max_pool3d(norm,'pool5')
    layers = [pool1,pool2,pool3,pool4,pool5]
    return layers[-1]
