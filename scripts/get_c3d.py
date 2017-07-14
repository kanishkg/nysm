import numpy as np
import tensorflow as tf
from temporal import temp_encoder
import h5py
from c3d_generator import batch_generator

input = tf.placeholder(shape = (1,16,256,480,3),dtype = tf.float32)
output = tf.contrib.layers.flatten(temp_encoder(input))

f= h5py.File('/scratch/kvg245/youtube_videos/c3d.h5','a')
bg = batch_generator(1)
bg.current_epoch = 0
bg.batch_index = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while bg.current_epoch == 0:
        inp , arr = bg.get_batch_vec()
        o = sess.run(output,feed_dict = {input:inp})
        print str(arr[0][0])+','+str(arr[0][2])
        f[str(arr[0][0])+','+str(arr[0][2])] = o
