import os
import random
import time
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
from metrics import get_hit_rate
from tsing_bg import batch_generator
from model import model
import matplotlib.pyplot as plt

mode = 'train'
max_epoch = 3
batch_size = 64
output_dir = '/scratch/kvg245/tsing_data/output/frame_16_MSE/'
seed = 4
ckpt = False

save_freq = 6400
val_freq = 1
stat_freq = 6400

if __name__ == "__main__":
    print "begin"
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
            loss = Model.train(batch['record'],batch['target'])
            print(i,loss)
            if i%100 == 0:
                pred = Model.forward(batch['record'],batch['target'])
                print (batch['target'][0,1,:])
                print (pred[0][0,1,:])
                print (batch['record'][0,1,:])

    elif mode == 'train':

        bg = batch_generator(batch_size)

        bg.current_epoch = 0
        bg.batch_index = 0
        Model = model(batch_size,ckpt = ckpt,output_dir = output_dir)
        avg_loss = 0
        i = 0

        def val():
            bg_val = batch_generator(batch_size,istrain = False)
            bg_val.current_epoch = 0
            bg_val.batch_index = 0
            batch = bg_val.get_batch_vec()
            loss = [0,0,0]
            iteration = 0
            while bg_val.current_epoch!=1:
                loss_current = Model.forward(batch['record'],
                                     batch['target'])
                summaries = loss_current[-1]
                loss_current = loss_current[:-1]
                loss = [x + y for x, y in zip(loss_current[1:],loss)]
                batch = bg_val.get_batch_vec()
                iteration+=1
                if bg_val.batch_index%stat_freq==0:
                    loss_print = [x / iteration for x in loss]
                    print "val_loss" , bg_val.batch_index, loss_print
            return loss[0]/iteration
        val_loss =0
        done = 0
        while bg.current_epoch<max_epoch:

            batch = bg.get_batch_vec()
            loss = Model.train(batch['record'],batch['target'])
            summary = loss[-1]
            avg_loss+=loss[0]
            i+=1

            if bg.batch_index%stat_freq ==0:
                print(bg.batch_index,bg.current_epoch,loss[:-1],avg_loss/i,val_loss)

            if bg.batch_index%save_freq==0:
                Model.save()

            if bg.batch_index%1000 ==0:
                i = 0
                avg_loss = 0
            if bg.current_epoch %val_freq ==0 and bg.current_epoch!=done:
                val_loss = val()
                done = bg.current_epoch
            Model.train_writer.add_summary(summary,bg.batch_index+bg.current_epoch*(bg.batch_len/bg.batch_size)*bg.batch_size)

    elif mode == 'val':

        Model = model(batch_size,ckpt =ckpt,output_dir=output_dir)
        l = 0

        for i in range(0,18):
            bg_val = batch_generator(batch_size,calc_vid=[i],istrain = False)
            bg_val.current_epoch = 0
            bg_val.batch_index = 0
            batch = bg_val.get_batch_vec()
            loss = [0,0,0]
            iteration = 0
            # means_theta = []
            # means_phi = []
            # std_theta = []
            # std_phi = []
            # hit_rate_1 = []
            # hit_rate_2 = []
            # hit_rate_3 = []
            while bg_val.current_epoch!=1:
                loss_current = Model.forward(batch['record'],
                                     batch['target'])
                summaries = loss_current[-1]
                loss_current = loss_current[:-1]
                out = loss_current[0]
                #print get_hit_rate(out,batch,1.0)
                # hit_rate_1 = hit_rate_1 +list(get_hit_rate(out,batch,1.0))
                # hit_rate_2 = hit_rate_2 + list(get_hit_rate(out,batch,1.5))
                # hit_rate_3 = hit_rate_3 + list(get_hit_rate(out,batch,2.0))
                # means_theta = means_theta+list(np.square(out[0][:,0]-np.mean(batch['target'][:,:20])))
                # means_phi = means_phi+list(np.square(out[0][:,1]-np.mean(batch['target'][:,20:])))
                # std_theta = std_theta + list(np.square(np.exp(out[1][:,0])-np.square(np.std(batch['target'][:,:20]))))
                # std_phi = std_phi + list(np.square(np.exp(out[1][:,1])-np.square(np.std(batch['target'][:,20:]))))
                loss = [x + y for x, y in zip(loss_current[1:],loss)]
                batch = bg_val.get_batch_vec()
                iteration+=1
                loss_print = [x / iteration for x in loss]
            print i,loss_print

            # plt.title('Video'+str(i))
            # plt.figure(i)
            # plt.subplot(411)
            # plt.hist(means_theta,normed=True, bins=360)
            # plt.xlabel('MSE_theta')
            # plt.ylabel('value')
            # plt.subplot(412)
            # plt.hist(means_phi,normed=True, bins=180)
            # plt.xlabel('MSE_phi')
            # plt.ylabel('value')
            # plt.subplot(413)
            # plt.hist(std_theta,normed=True, bins=360)
            # plt.xlabel('std_theta')
            # plt.ylabel('value')
            # plt.subplot(414)
            # plt.hist(std_phi,normed=True, bins=180)
            # plt.xlabel('std_phi')
            # plt.ylabel('value')

            # plt.savefig('result_plots/'+str(i)+'.png')

            l+=loss_print[0]
        print l/18.0
