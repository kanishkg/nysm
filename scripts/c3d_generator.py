import pickle
import numpy as np
from utils import shuffled
import random
import h5py
import time

class batch_generator:

    def __init__( self,batch_size = 8,istrain = True, input_file = '/scratch/kvg245/youtube_videos/video_dataset/video.h5',target_file = '/scratch/kvg245/youtube_videos/spherical_coordinates.npy',num_frames = 16,index_file = '/scratch/kvg245/youtube_videos/index_c3d3.npy',seq_length = 20,rec_length = 20):
        self.batch_size = batch_size
        self.istrain = istrain
        self.seq_length = seq_length
        self.rec_length = rec_length
        random.seed(4)
        self.val_list = list(random.sample(range(0,57),int(57*0.2)))
        self.index_data,self.target_data,self.input_data = self.open_files(target_file,input_file,index_file)
        self.batch_len = len(self.index_data)
        self.current_epoch = None
        self.batch_index = None
        self.num_frames = num_frames
        self.video = None
        self.fps  = [30,30,29.97,25,60,30]

    def open_files(self,target_file,input_file,index_file):

        input = h5py.File(input_file,'r')
        target =  np.load(target_file)

        index_data = np.load(index_file)
        i = index_data[1]
        index_data[0] = i
        print len(index_data)

        return index_data,target, input

    def create_batch(self, data_list):
        """Creates and returns a batch of input and output data"""

        input_batch = []
        ind = sorted([-3,-5,-8,-10,-12,-15,-17,-20,0,3,6,9,12,15,17,19])
        for data in data_list:
            arr = [int(data[2] +int(i*self.fps[data[0]]/20.0)) for i in range(-20,20) if i in ind]
            video = (self.input_data[str(data[0])][arr,...])
            input_batch.append(video)

        input_batch = np.asarray(input_batch)
        return [input_batch, data_list]

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
    print(a[0].shape,a[1])
