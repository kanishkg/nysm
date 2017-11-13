import pickle
from math import pi
import numpy as np
from utils import shuffled
import random
import h5py
import time
import math
import pandas as pd

class batch_generator:

    def __init__( self,batch_size = 8,istrain = True, calc_vid=[0,11,3,13],
                 target_file = '/scratch/kvg245/tsing_data/Formated_Data/data.npy',num_frames = 16,
                 index_file = '/scratch/kvg245/tsing_data/Formated_Data/index.npy',seq_length = 20,rec_length = 15):
        self.batch_size = batch_size
        self.istrain = istrain
        self.seq_length = seq_length
        self.rec_length = rec_length
        self.calc_vid = calc_vid
        random.seed(4)
        self.val_list = list(random.sample(range(0,47),int(48*0.2)))
        self.index_data,self.target_data = self.open_files(target_file,index_file)
        self.batch_len = len(self.index_data)
        self.current_epoch = None
        self.batch_index = None
        self.num_frames = num_frames
        self.video = None
        self.fps  = list(pd.read_csv(
                '/scratch/kvg245/tsing_data/Formated_Data/Experiment_1/videoMeta.csv')['FrameRate'].values)

        self.fps  = self.fps+list(pd.read_csv(
                '/scratch/kvg245/tsing_data/Formated_Data/Experiment_2/videoMeta.csv')['FrameRate'].values)

    def open_files(self,target_file,index_file):

        target =  np.load(target_file)

        index_data = np.load(index_file)
        if self.istrain:
            index_data = shuffled([a for a in index_data if a[1] not in self.calc_vid and a[1]!=0 and a[0] not in self.val_list])
        else:
            index_data = [a for a in index_data if a[1] in self.calc_vid and a[0] in self.val_list]

        print len(index_data)

        return index_data,target

    def create_batch(self, data_list):
        """Creates and returns a batch of input and output data"""

        input_batch = []
        record_batch = []
        target_batch = []
        for data in data_list:
            target = []
            record = []
            video = []
            for seq in range(self.seq_length):
                #target.append(np.array([self.target_data[data[0],data[1]][int(data[2] +int(i*self.fps[data[1]]/15.0))][2:] for i in range(15)]))
                record.append(np.array([self.target_data[data[0],data[1]][int(data[2]-seq*self.fps[data[1]] -int(i*self.fps[data[1]]/15.0)-1)][2:] for i in range(15)]))
            #target_batch.append(target[-1])
            target_batch.append(np.array([self.target_data[data[0],data[1]][int(data[2] +int(i*self.fps[data[1]]/15.0))][2:] for i in range(15)]))
            record_batch.append(record)

        target_batch = np.asarray(target_batch)
        record_batch = np.asarray(record_batch)
        target = np.zeros((self.batch_size,self.rec_length*2))
        record = np.zeros((self.batch_size,self.seq_length,self.rec_length*2))
        target[:,:self.rec_length] = (target_batch[:,:,0]+pi)/(2*pi)
        record[:,:,:self.rec_length] = (record_batch[:,:,:,0]+pi)/(2*pi)
        target[:,self.rec_length:] = target_batch[:,:,1]/pi
        record[:,:,self.rec_length:] = record_batch[:,:,:,1]/pi

        return { 'target':np.nan_to_num(target), 'record':np.nan_to_num(record) }

    def get_batch_vec(self):
        """Provides batch of data to process and keeps
        track of current index and epoch"""

        if self.batch_index is None:
            self.batch_index = 0
            self.current_epoch = 0

        if self.batch_index < self.batch_len-self.batch_size-1:
            batch_dict = self.create_batch(self.index_data[self.batch_index:self.batch_index + self.batch_size])
            self.batch_index += self.batch_size
        else:
            self.current_epoch += 1
            self.batch_index = 0
            batch_dict = self.create_batch(self.index_data[self.batch_index:self.batch_index + self.batch_size])
            self.batch_index += self.batch_size

        return batch_dict


if __name__ == "__main__":
    bg = batch_generator(1)
    a = bg.get_batch_vec()
    print a
    while bg.current_epoch ==0:
        break
        a = bg.get_batch_vec()
        print bg.batch_index

