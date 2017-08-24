import os
import random
import time

import tensorflow as tf
import numpy as np

from batch_generator import batch_generator
from model import model


mode = 'overfit'
max_epoch = 3000
batch_size = 16
output_dir = '/scratch/kvg245/youtube_videos/output/train1/'
seed = 4
ckpt = False

save_freq = 4000

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
            if i%100 == 0:
                pred = Model.forward(batch['input'],batch['record'],batch['target']) 
                print (batch['target'][0,0,:])
                print (pred[0,0,:])
                print (batch['record'][0,0,:])
    elif mode == 'train':

        bg =batch_generator(batch_size)
        bg.current_epoch = 0
        bg.batch_index = 0
        Model = model(batch_size,ckpt = ckpt,output_dir = output_dir)
        avg_loss = 0
        i = 0

        while bg.current_epoch<max_epoch:
            start = time.time()
            batch = bg.get_batch_vec()
            compute = time.time()
            loss = Model.train(batch['input'],batch['record'],batch['target'])
            avg_loss+=loss
            i+=1
            end = time.time()
            print(bg.batch_index,bg.current_epoch,loss,avg_loss/i,compute-start,end-compute,(end-start)*(bg.batch_len-bg.batch_index)*(max_epoch-bg.current_epoch-1))

            if bg.batch_index%save_freq==0:
                Model.save()

            if bg.batch_index%1000 ==0:
                i = 0
                avg_loss = 0

