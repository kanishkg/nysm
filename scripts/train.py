import tensorflow as tf
import numpy as np
import os
import random

from batch_generator import batch_generator
from model import model


mode = 'overfit'
max_epoch = 300
batch_size = 4
output_dir = '/scratch/kvg245/youtube_videos/output/test1/'
seed = 4
ckpt = False

if __name__ == "__main__":

    if not os.path.exists(output_dir):
         os.makedirs(output_dir)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if mode == 'overfit':
        bg = batch_generator(batch_size)
        Model = model(batch_size,ckpt = ckpt,output_dir = output_dir)
        batch = bg.get_batch_vec()
        for i in range(max_epoch):
            loss = Model.train(batch['input'],batch['record'],batch['target'])
            print(i,loss)

