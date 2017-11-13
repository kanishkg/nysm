import tensorflow as tf
import numpy as np
from utils import *
from math import pi


class model(object):

    def __init__(self, batch_size=1, seq_length=20, rec_length=15, ckpt=False, output_dir=''):
        self.seq_length = seq_length
        self.rec_length = rec_length
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.input_record = tf.placeholder(
            shape=(batch_size, seq_length, rec_length * 2), dtype=tf.float32)
        self.target = tf.placeholder(
            shape=(batch_size, rec_length * 2), dtype=tf.float32)

        with tf.variable_scope("fully_connected"):
            self.lstm_input = self.input_record

        with tf.variable_scope("LSTM"):
            self.cell = tf.nn.rnn_cell.LSTMCell(
                num_units=16, state_is_tuple=True)
            self.outputs, _ = tf.nn.dynamic_rnn(
                self.cell, self.lstm_input, dtype=tf.float32)
            print self.outputs[:, -1, :]

        with tf.variable_scope("output"):
            self.outputs = tf.sigmoid(tf.contrib.layers.fully_connected(
                self.outputs[:, -1, :], rec_length * 2, activation_fn=None))

        with tf.variable_scope("loss"):
            self.loss = self.def_loss(self.outputs, self.target)

        self.sess = tf.Session()
        self.optimizer = tf.train.AdamOptimizer(0.001)
        self.variables = [var for var in tf.trainable_variables()]
        self.grads = self.optimizer.compute_gradients(
            self.loss, var_list=self.variables)
        self.apply_train = self.optimizer.apply_gradients(self.grads)
        self.saver = tf.train.Saver()
        self.restore_saver = tf.train.Saver()

        if ckpt:
            print("loading model from checkpoint")
            print(output_dir)
            checkpoint = tf.train.latest_checkpoint(output_dir)
            self.restore_saver.restore(self.sess, checkpoint)
        else:
            self.initialize()

        with tf.name_scope('summaries'):
            tf.summary.scalar('total_loss', self.loss)
            tf.summary.scalar('MSE_theta', self.MSE_theta)
            tf.summary.scalar('MSE_phi', self.MSE_phi)
        self.merged_summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(output_dir,
                                                  self.sess.graph)

    def save(self):
        self.saver.save(self.sess, self.output_dir + "model.ckpt")

    def def_loss(self, out, target):
        #loss = tf.reduce_sum(tf.square(target - out))/(self.batch_size*self.seq_length*self.rec_length*2)
        self.MSE_theta = tf.reduce_mean(tf.minimum(tf.square(target[:, :, :self.rec_length] * 2 * pi - 2 * pi * out[:, :, :self.rec_length]),
                                                   tf.square(2 * pi - 2 * pi * target[:, :, :self.rec_length] - 2 * pi * out[:, :, :self.rec_length])))
        self.MSE_phi = tf.reduce_mean(
            tf.square(target[:, :, self.rec_length:] * pi - pi * out[:, :, self.rec_length:]))
        # self.theta_log_likelihood =tf.reduce_mean(tf.div(tf.minimum(tf.square(
        #     target[:,:self.rec_length]*2*pi-2*pi*tf.tile(tf.expand_dims(means[:,0],-1),[1,self.rec_length])),
        #     tf.square(2*pi-tf.abs(2*pi*target[:,:self.rec_length]-2*pi*tf.tile(tf.expand_dims(means[:,0],-1),[1,self.rec_length])))),tf.tile(tf.expand_dims(sigmas[:,0],-1),[1,self.rec_length])))
#        self.theta_log_likelihood =tf.reduce_mean(tf.div(tf.square(
#            target[:,:self.rec_length]*2*pi-2*pi*tf.tile(tf.expand_dims(means[:,0],-1),[1,self.rec_length])),
#            tf.tile(tf.expand_dims(sigmas[:,0],-1),[1,self.rec_length])))

        # self.phi_log_likelihood =tf.reduce_mean(tf.div(tf.square(
        #     target[:,self.rec_length:]*pi-pi*tf.tile(tf.expand_dims(means[:,1],-1),[1,self.rec_length])),tf.tile(tf.expand_dims(sigmas[:,1],-1),[1,self.rec_length])))
        # self.MSE_theta = tf.reduce_mean(tf.minimum(tf.square(
        #     target[:,:self.rec_length]*2*pi-2*pi*tf.tile(tf.expand_dims(means[:,0],-1),[1,self.rec_length])),
        #     tf.square(2*pi-tf.abs(2*pi*target[:,:self.rec_length]-2*pi*tf.tile(tf.expand_dims(means[:,0],-1),[1,self.rec_length])))))
#         self.MSE_theta = tf.reduce_mean(tf.square(
#             target[:,:self.rec_length]*2*pi-2*pi*tf.tile(tf.expand_dims(means[:,0],-1),[1,self.rec_length])))

        # self.MSE_phi = tf.reduce_mean(tf.square(
        #     target[:,self.rec_length:]*pi-pi*tf.tile(tf.expand_dims(means[:,1],-1),[1,self.rec_length])))
        # self.theta_log_likelihood += tf.reduce_mean(sigma[:,0])
        # self.phi_log_likelihood += tf.reduce_mean(sigma[:,1])
        # self.sigmas_theta = tf.reduce_mean(sigmas[:,0])
        # self.sigmas_phi = tf.reduce_mean(sigmas[:,1])
        # NLL = self.theta_log_likelihood+self.phi_log_likelihood
        loss = self.MSE_phi + self.MSE_theta
        return loss

    def initialize(self):
        print("initialize")
        self.sess.run(tf.global_variables_initializer())

    def forward(self, record_array, target_array):
        out = self.sess.run([self.outputs, self.loss,
                             self.MSE_theta,
                             self.MSE_phi, self.merged_summaries],
                            {self.input_record: record_array, self.target: target_array})
        return out

    def train(self, record_array, target_array):
        loss = self.sess.run([self.loss,
                              self.MSE_theta,
                              self.MSE_phi, self.merged_summaries, self.apply_train], {self.target: target_array, self.input_record: record_array})
        return loss[:-1]


if __name__ == "__main__":
    Model = model(8)
    b = np.ones((8, 20, 30))
    c = np.ones((8, 30))
    _ = Model.train(b, c)
