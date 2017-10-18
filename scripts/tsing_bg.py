import pickle
import numpy as np
from utils import shuffled
import random
import h5py
import time
import math

class batch_generator:

    def __init__( self,batch_size = 8,istrain = True, calc_vid=None,
                 target_file = '/scratch/kvg245/tsing_data/Formated_Data/data.npy',num_frames = 16,
                 index_file = '/scratch/kvg245/tsing_data/Formated_Data/index.npy',seq_length = 20,rec_length = 20):
        self.batch_size = batch_size
        self.istrain = istrain
        self.seq_length = seq_length
        self.rec_length = rec_length
        self.calc_vid = calc_vid
        random.seed(4)
        self.val_list = list(random.sample(range(0,18),int(18*0.2)))
        if not frozen:
            self.index_data,self.target_data = self.open_files(target_file,index_file)
        else:
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
            index_data = shuffled([a for a in index_data if a[1] not in self.val_list and a[1]!=0])
        else:
            index_data = [a for a in index_data if a[1] in self.val_list and a[1]!=0]
        if self.calc_vid not None:
            index_data = [a for a in index_data if a[1]==self.calc_vid]


        print len(index_data)

        return index_data,target

    def create_batch(self, data_list):
        """Creates and returns a batch of input and output data"""

        input_batch = []
        record_batch = []
        target_batch = []
        ind = sorted([-3,-5,-8,-10,-12,-15,-17,-20,0,3,6,9,12,15,17,19])
        for data in data_list:
            target = []
            record = []
            video = []
            for seq in range(self.seq_length):
                target.append(self.target_data[data[0]][data[1],[int(data[2]+seq*self.fps[data[0]] +int(i*self.fps[data[0]]/20.0)) for i in range(20)], 2:])
                record.append(self.target_data[data[0]][data[1],[int(data[2]+seq*self.fps[data[0]] -int(i*self.fps[data[0]]/20.0)-1) for i in range(20)],2:])
                arr = [int(data[2]+seq*self.fps[data[0]] +int(i*self.fps[data[0]]/20.0)) for i in range(-20,20) if i in ind]
            target_batch.append(target[-1])
            record_batch.append(record)

        target_batch = np.asarray(target_batch)
        record_batch = np.asarray(record_batch)
        target = np.zeros((self.batch_size,20*2))
        # target = target_batch
        record = np.zeros((self.batch_size,self.seq_length,20*2))
        target[:,:20] = target_batch[:,:,0]
        record[:,:,:20] = record_batch[:,:,:,0]
        target[:,20:] = target_batch[:,:,1]
        record[:,:,20:] = record_batch[:,:,:,1]

        return { 'target':target, 'record':record }

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
    bg = batch_generator(4,frozen = True)
    a = bg.get_batch_vec()
    while bg.current_epoch ==0:

        a = bg.get_batch_vec()
        print bg.batch_index

