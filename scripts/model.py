import tensorflow as tf
import numpy as np
from utils import *
from math import pi

class model(object):

    def __init__(self, batch_size = 1, seq_length=20,rec_length=20,ckpt = False,output_dir = '',frozen = True):
        self.seq_length = seq_length
        self.rec_length = rec_length
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.frozen = frozen

        self.input_tensor = tf.placeholder(shape = (batch_size, seq_length,16,256,480,3),dtype = tf.float32)
        self.feat_tensor = tf.placeholder(shape = (batch_size,seq_length,61440),dtype=tf.float32)
        self.input_record = tf.placeholder(shape=(batch_size,seq_length,rec_length*2),dtype = tf.float32)
        self.target = tf.placeholder(shape = (batch_size, seq_length,rec_length*2),dtype = tf.float32)
        # self.target = tf.placeholder(shape = (batch_size, seq_length,4),dtype = tf.float32)
        with tf.variable_scope("temp_enc"):
            if not frozen:
                function = lambda x: tf.contrib.layers.flatten(self.temporal_encoder(x)[-1])
                self.temp_enc = tf.map_fn(function,tf.transpose(self.input_tensor,[1,0,2,3,4,5]),parallel_iterations = 1,swap_memory =True,back_prop = False)
                function = lambda x: tf.contrib.layers.fully_connected(x,4096)
                self.fc_enc = tf.map_fn(function,self.temp_enc)
                self.temp_enc = tf.transpose(self.fc_enc,[1,0,2])
            else:
                pass
                #function = lambda x: tf.contrib.layers.fully_connected(x,4096)
                #self.fc_enc = tf.map_fn(function,tf.transpose(self.feat_tensor,[1,0,2]))
                #self.temp_enc = tf.transpose(self.fc_enc,[1,0,2])

        with tf.variable_scope("fully_connected"):
            #self.lstm_input = tf.concat([self.temp_enc,self.input_record],axis = 2)
            self.lstm_input = self.input_record

        with tf.variable_scope("LSTM"):
            self.cell = tf.nn.rnn_cell.LSTMCell(num_units=16, state_is_tuple=True)
            self.outputs, _ = tf.nn.dynamic_rnn(self.cell,self.lstm_input,dtype = tf.float32)

        with tf.variable_scope("output"):
            # self.outputs = tf.sigmoid(tf.contrib.layers.fully_connected(self.outputs,rec_length*2,activation_fn = None))
            self.outputs = tf.sigmoid(tf.contrib.layers.fully_connected(self.outputs,
                                                                        4,activation_fn = None))


        with tf.variable_scope("loss"):
            self.loss = self.def_loss(self.outputs, self.target)


        self.sess = tf.Session()

        self.optimizer = tf.train.AdamOptimizer(0.001)
        self.variables = [var for var in tf.trainable_variables()]
        self.grads = self.optimizer.compute_gradients(self.loss, var_list = self.variables)
        self.apply_train = self.optimizer.apply_gradients(self.grads)
        self.saver = tf.train.Saver()
        self.restore_saver = tf.train.Saver()

        if ckpt:
            print("loading model from checkpoint")
            print(output_dir)
            checkpoint = tf.train.latest_checkpoint(output_dir)
            self.restore_saver.restore(self.sess,checkpoint)
        else:
            self.initialize()

    def save(self):
	    self.saver.save(self.sess, self.output_dir+"model.ckpt")

    def temporal_encoder(self,inputs):
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
        with tf.device('/gpu:1'):
    	    with tf.variable_scope('encoder3d_3'):
                conv3_1 = conv3d_layer2(data_dict,pool2, 128, 256, "conv3_1")
                conv3_2 = conv3d_layer2(data_dict,conv3_1, 256, 256, "conv3_2")
                norm = tf.contrib.slim.batch_norm(conv3_2)
                pool3 = max_pool3d(norm, 'pool3')

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
    	return layers


    def def_loss(self, out, target):
        #loss = tf.reduce_sum(tf.square(target - out))/(self.batch_size*self.seq_length*self.rec_length*2)
        # loss = tf.reduce_mean(tf.minimum(tf.square(target[:,:,:self.rec_length]*2*pi-2*pi*out[:,:,:self.rec_length]),
        #     tf.square(2*pi-2*pi*target[:,:,:self.rec_length]-2*pi*out[:,:,:self.rec_length])))
        # loss +=tf.reduce_mean(
        #     tf.square(target[:,:,self.rec_length:]*pi-pi*out[:,:,self.rec_length:]))
        sigmas = tf.exp(out[:,:,2:])
        means = out[:,:,:2]
        self.theta_log_likelihood =tf.div(tf.minimum(tf.square(
            target[:,:,:self.rec_length]*2*pi-2*pi*tf.tile(means[:,:,0],self.rec_length)),
            tf.square(2*pi-2*pi*target[:,:,:self.rec_length]-2*pi*tf.tile(means[:,:,0],self.rec_length))),tf.tile(sigmas[:,:,0],self.rec_length))
        self.phi_log_likelihood =tf.div(tf.square(
            target[:,:,:self.rec_length]**pi-*pi*tf.tile(means[:,:,1],self.rec_length)),tf.tile(sigmas[:,:,1],self.rec_length))
        self.MSE_theta = tf.minimum(tf.square(
            target[:,:,:self.rec_length]*2*pi-2*pi*tf.tile(means[:,:,0],self.rec_length)),
            tf.square(2*pi-2*pi*target[:,:,:self.rec_length]-2*pi*tf.tile(means[:,:,0],self.rec_length)))
        self.MSE_phi = tf.square(
            target[:,:,:self.rec_length]**pi-*pi*tf.tile(means[:,:,1],self.rec_length)),tf.tile(sigmas[:,:,1],self.rec_length))
        NLL = self.theta_log_likelihood+self.phi_log_likelihood
        return NLL

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())


    def forward(self, input_array,record_array, target_array):
        out = self.sess.run([self.outputs,self.loss,self.theta_log_likelihood,
                             self.phi_log_likelihood, self.MSE_theta,
                             self.MSE_phi],
                            {self.feat_tensor : input_array,self.input_record:record_array,self.target : target_array})
        return out


    def train(self, input_array, record_array,target_array):
        if not self.frozen:
            loss, _ = self.sess.run([self.loss, self.apply_train], {self.input_tensor: input_array,
                self.target: target_array, self.input_record: record_array})
        else:
            loss, _ = self.sess.run([self.loss, self.apply_train], {self.feat_tensor: input_array,self.target: target_array, self.input_record: record_array})
        return loss


if __name__ == "__main__":
    Model = model(8)
    a = np.zeros((8,20,61440))
    b = np.zeros((8,20,40))
    c= np.ones((8,20,40))
    a = Model.train(a,b,c)
