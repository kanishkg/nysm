import tensorflow as tf
import numpy as np
from utils import *
from math import pi

class model(object):

    def __init__(self, batch_size = 1, seq_length=20,rec_length=20,ckpt = False,output_dir = ''):
        self.seq_length = seq_length
        self.rec_length = rec_length
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.input_record = tf.placeholder(shape=(batch_size,seq_length,rec_length*2),dtype = tf.float32)
        self.target = tf.placeholder(shape = (batch_size, seq_length,rec_length*2),dtype = tf.float32)

        with tf.variable_scope("fully_connected"):
            self.lstm_input = self.input_record

        with tf.variable_scope("LSTM"):
            self.cell = tf.nn.rnn_cell.LSTMCell(num_units=16, state_is_tuple=True)
            self.outputs, _ = tf.nn.dynamic_rnn(self.cell,self.lstm_input,dtype = tf.float32)

        with tf.variable_scope("output"):
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
            self.initialize








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
            target[:,:,:self.rec_length]*pi-pi*tf.tile(means[:,:,1],self.rec_length)),tf.tile(sigmas[:,:,1],self.rec_length))
        self.MSE_theta = tf.reduce_mean(tf.minimum(tf.square(
            target[:,:,:self.rec_length]*2*pi-2*pi*tf.tile(means[:,:,0],self.rec_length)),
            tf.square(2*pi-2*pi*target[:,:,:self.rec_length]-2*pi*tf.tile(means[:,:,0],self.rec_length))))
        self.MSE_phi = tf.reduce_mean(tf.square(
            target[:,:,self.rec_length:]*pi-pi*tf.tile(means[:,:,1],self.rec_length)))
        NLL = self.theta_log_likelihood+self.phi_log_likelihood
        return NLL

    def initialize(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('total_loss',self.loss)
            tf.summary.scalar('MSE_theta', self.MSE_theta)
            tf.summary.scalar('MSE_phi', self.MSE_theta)
            tf.summary.scalar('phi_log_likelihood', self.phi_log_likelihood)
            tf.summary.scalar('theta_log_likelihood', self.theta_log_likelihood)
        self.merged_summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(output_dir ,
                                      self.sess.graph)

        self.sess.run(tf.global_variables_initializer())


    def forward(self, record_array, target_array):
        out = self.sess.run([self.outputs,self.loss,self.theta_log_likelihood,
                             self.phi_log_likelihood, self.MSE_theta,
                             self.MSE_phi],
                            {self.feat_tensor : input_array,self.input_record:record_array,self.target : target_array})
        return out

    def train(self,  record_array,target_array):
        else:
            loss,summaries, _ = self.sess.run([self.loss, self.merged_summaries, self.apply_train], {self.target: target_array, self.input_record: record_array})
        return loss,summaries


if __name__ == "__main__":
    Model = model(8)
    b = np.zeros((8,20,40))
    c= np.ones((8,20,40))
    a = Model.train(a,b,c)
