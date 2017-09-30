import tensorflow as tf
import numpy as np
from utils import *
from math import pi

class model(object):

    def __init__(self, batch_size = 1, seq_length=20,ckpt = False,output_dir = ''):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.output_dir = output_dir

        self.input_record =
        tf.placeholder(shape=(batch_size,seq_length,2),dtype = tf.float32)
        self.target = tf.placeholder(shape = (batch_size,
                                              seq_length,2),dtype = tf.float32)


        with tf.variable_scope("encoder"):
            self.cell = tf.nn.rnn_cell.BasicRNNCell(num_units=8)
            self.outputs, self.states = tf.nn.dynamic_rnn(self.cell,self.lstm_input,dtype = tf.float32)

        with tf.variable_scope("decoder"):
#            self.cell = tf.nn.rnn_cell.LSTMCell(num_units = 16,
#                                                state_is_tuple = True)
#            helper =
#            tf.contrib.seq2seq.TrainingHelper(input=input_record,sequence_length=seq_length)
#            decoder = tf.contrib.seq2seq.Decoder(self.cell,helper,
#                    initial_state =self.cell.zero_state(batch_size,dtype=
#                                                        tf.float32)
#            self.outputs, _ = tf.contrib.seq2seq.dynamic_decode(
#                               decoder=decoder,
#                               output_time_major=False,
#                               impute_finished=True,
#                               maximum_iterations=20)
            o = self.outputs[-1]
            s = self.states[-1]
            self.outputs = []
            for i in range(self.seq_length):
                x = tf.sigmoid(tf.contrib.layers.fully_connected(o,2))
                self.outputs.append(x)
                o = self.cell(x,s)
                s = o
            self.outputs = tf.stack(self.outputs)
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
            restore_saver.restore(sess,checkpoint)
        else:
            self.initialize()

    def save(self):
	    self.saver.save(self.sess, self.output_dir+"model.ckpt")


    def def_loss(self, out, target):
        loss = tf.reduce_sum(tf.square(target - out))/(self.batch_size*self.seq_length*self.rec_length*2)
        #loss = tf.reduce_mean(tf.minimum(tf.square(target[:,:,:self.rec_length]*2*pi-2*pi*out[:,:,:self.rec_length]),
        #    tf.square(2*pi-2*pi*target[:,:,:self.rec_length]-2*pi*out[:,:,:self.rec_length])))
        #loss += 4*tf.reduce_mean(tf.square(target[:,:,:self.rec_length]*pi-pi*out[:,:,:self.rec_length]))
        return loss

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())


    def forward(self, record_array, target_array):
        out = self.sess.run(self.outputs, {input_record:record_array,self.target : target_array})
        return out


    def train(self, record_array,target_array):
        loss, _ = self.sess.run([self.loss, self.apply_train], {
                self.target: target_array, self.input_record: record_array})
        return loss


if __name__ == "__main__":
    Model = model(8)
    b = np.zeros((8,20,2))
    c= np.ones((8,20,2))
    a = Model.train(b,c)
