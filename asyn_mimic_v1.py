#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Xiao Jin
In this file we construct async nonlinear part on mimic3 dataset
"""

from functools import partial
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Barrier
import numpy as np
import tensorflow as tf
import random as r
import copy as cp
from config_v1 import *
from time import time, sleep
from roc_plot import ROC_curve
# import modelnet40 training data
# from modelnet_preprocess_v1 import async_train_dataset, test_dataset
import model_LSTM_v1 as model

# load data
train_data = np.load('train_data.npy')
train_label = np.load('train_label.npy')
test_data = np.load('test_data.npy')
test_label = np.load('test_label.npy')
num_train_samples = train_data.shape[0]
num_test_samples = test_data.shape[0]


train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_data, train_label))
    .batch(num_train_samples).shuffle(buffer_size=num_train_samples, seed=0)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices((tf.cast(test_data, tf.float32), test_label)).batch(num_test_samples)
)


def delay(coe, index):
    t = np.random.exponential(index + 1) * coe
    sleep(t)


def define_local_model(x, workers):
    # with tf.device('/job:localhost/replica:0/task:0/device:GPU:' + str(x)):
    with tf.device('/job:localhost/replica:0/task:0/device:GPU:0'):
        # initialize local models
        workers[x] = model.local_embedding()


class system():
    def __init__(self):
        self.seed = tf.random.set_seed(10)
        self.workers = {}
        """ initialize 4 local embeddings in the parallel way"""
        self.pool = Pool(2 * num_workers)
        self.mask = model.masking()
        """ initialize 4 workers"""
        self.pool.map(partial(define_local_model, workers=self.workers), list(range(num_workers)))  # [0, 1, 2, 3])
        """ initialize the server """
        self.server = model.server()
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = [tf.keras.optimizers.Adam(learning_rate=learning_rate)
                          for i in range(num_workers)]
        self.optimizer_s = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # self.optimizer_w = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.train_cost = tf.keras.metrics.Mean(name='train_cost')
        self.test_cost = tf.keras.metrics.Mean(name='test_cost')
        # self.w = tf.Variable(tf.random.normal((num_views*num_workers, ))/num_views/num_workers , name = "w")
        self.nob = int(num_train_samples / batch_size)
        self.V = np.full((num_train_samples, num_units, num_workers), 0.0)
        self.begin = False
        """ shared memory"""
        # self.idlist = tf.range(num_train_samples)
        # self.local_gradients = []
        """ initialize barrier for synchronization """
        # self.before_shuffle = Barrier(num_workers)
        # self.after_shuffle = Barrier(num_workers)
        # self.before_agg = Barrier(num_workers)
        # self.after_server = Barrier(num_workers)

    def asyn(self, index, train_dataset, test_dataset):
        f = open('asyn_mimic_lstm_v1' + '.txt', 'w')
        f.close()
        iterations = 0
        # training_loss = 5
        # training_accuracy = 0
        # total_accuracy = 0
        previous_time = 0
        epoch = 0
        train_x, train_y = zip(*train_dataset)
        train_x = train_x[0]
        train_y = train_y[0]
        # masking
        train_x = self.mask.forward(train_x)
        if index == 0:
            train_loss, train_accuracy, train_M, train_auc = self.test(index, train_dataset, 'vfl_async_train_roc')
            test_loss, test_accuracy, test_M, test_auc = self.test(index, test_dataset, 'vfl_async_test_roc')
            print(index, iterations, previous_time, train_loss, train_accuracy, train_auc, test_loss, test_accuracy,
                  test_auc)
            # writing training loss
            f = open('asyn_mimic_lstm_v1' + '.txt', 'a+')
            f.write(str(index))
            f.write('\t')
            f.write(str(iterations))
            f.write('\t')
            f.write(str(previous_time))
            f.write('\t')
            # writing training loss
            f.write(str(train_loss))
            f.write('\t')
            # writing training accuracy
            f.write(str(train_accuracy))
            f.write('\t')
            # writing training auc
            f.write(str(train_auc))
            f.write('\t')
            # writing testing loss
            f.write(str(test_loss))
            f.write('\t')
            # writing testing accuracy
            f.write(str(test_accuracy))
            f.write('\t')
            # writing testing auc
            f.write(str(test_auc))
            f.write('\n')
            f.close()
            self.begin = True
        while previous_time < threshold_time:
            if not self.begin:
                continue
            epoch += 1
            idlist = list(range(num_train_samples))
            r.shuffle(idlist)
            tf.dtypes.cast(idlist, dtype=tf.int32)
            running_loss = []
            running_accuracy = []
            for k in range(self.nob):
                # print(index)
                start = time()
                # print(k)
                ''' images and labels '''
                batchid = idlist[k * batch_size:(k + 1) * batch_size]
                vertical_data = []
                vertical_data.append(tf.gather(train_x[:, :, 0:12], batchid))
                vertical_data.append(tf.gather(train_x[:, :, 12:49], batchid))
                vertical_data.append(tf.gather(train_x[:, :, 49:59], batchid))
                vertical_data.append(tf.gather(train_x[:, :, 59:76], batchid))
                labels = tf.gather(train_y, batchid)

                ''' forward '''
                '''
                with tf.device('/job:localhost/replica:0/task:0/device:GPU:' + str(index)):
                '''
                with tf.GradientTape(persistent=True) as tape:
                    ''' local embedding '''
                    local_tmpout = self.workers[index].forward(vertical_data[index])
                    # batch x 16 x 1
                '''
                with tf.device('/job:localhost/replica:0/task:0/device:CPU:0'):
                '''
                # print(index, k)
                with tf.GradientTape(persistent=True) as tape_s:
                    self.V[batchid, :, index] = cp.copy(local_tmpout)
                    ''' server aggregation '''
                    local_outputs = tf.convert_to_tensor(self.V[batchid, :, :], dtype=tf.float32)
                    # batchsize x 16x 4
                    tape_s.watch(local_outputs)
                    new_input = tf.reshape(local_outputs, [-1, num_units * num_workers])
                    output = self.server.forward(new_input)
                    loss = self.loss(labels, output)
                    ''' keep track of loss and accuracy '''
                    training_loss = self.train_cost(loss).numpy()
                    training_accuracy = tf.reduce_mean(
                        tf.keras.metrics.sparse_categorical_accuracy(labels, output)).numpy()
                    running_loss.append(training_loss)
                    running_accuracy.append(training_accuracy)
                    M = tf.math.confusion_matrix(labels.numpy(), tf.math.round(output).numpy())
                    # print(index)
                    # print('=========>training:\n', M.numpy())

                ''' backward and optimize'''
                ''' server '''
                server_gradients = tape_s.gradient(loss, self.server.trainable_variables)
                self.optimizer_s.apply_gradients(zip(server_gradients, self.server.trainable_variables))
                # self.optimizer_w.apply_gradients(zip(w_gradients, [self.w]))
                self.local_gradients = tape_s.gradient(loss, local_outputs)
                ''' local worker '''
                worker_gradients = tape.gradient(local_tmpout, self.workers[index].trainable_variables,
                                                 output_gradients=self.local_gradients[:, :, index])
                self.optimizer[index].apply_gradients(zip(worker_gradients, self.workers[index].trainable_variables))
                delay(delay_scale, index)
                end = time()
                previous_time += end - start

                if iterations % 4 == index:
                    # print('loss')
                    train_loss, train_accuracy, train_M, train_auc = self.test(index, train_dataset, 'vfl_async_train_roc')
                    test_loss, test_accuracy, test_M, test_auc = self.test(index, test_dataset, 'vfl_async_test_roc')
                    '''
                    print(index, iterations, previous_time, train_loss, train_accuracy, train_auc, test_loss, test_accuracy,
                          test_auc)
                    '''
                    # writing training loss
                    f = open('asyn_mimic_lstm_v1' + '.txt', 'a+')
                    f.write(str(index))
                    f.write('\t')
                    f.write(str(iterations))
                    f.write('\t')
                    f.write(str(previous_time))
                    f.write('\t')
                    # writing training loss
                    f.write(str(train_loss))
                    f.write('\t')
                    # writing training accuracy
                    f.write(str(train_accuracy))
                    f.write('\t')
                    # writing training auc
                    f.write(str(train_auc))
                    f.write('\t')
                    # writing testing loss
                    f.write(str(test_loss))
                    f.write('\t')
                    # writing testing accuracy
                    f.write(str(test_accuracy))
                    f.write('\t')
                    # writing testing auc
                    f.write(str(test_auc))
                    f.write('\n')
                    f.close()
                iterations += 1

    def test(self, index, dataset, name):
        test_x, test_y = zip(*dataset)
        test_x = test_x[0]
        test_y = test_y[0]
        samples = test_x.shape[0]
        V = np.full((samples, num_units, num_workers), 0.0)
        test_x = self.mask.forward(test_x)
        # with tf.device('/job:localhost/replica:0/task:0/device:GPU:' + str(index)):
        vertical_data = []
        vertical_data.append(test_x[:, :, 0:12])
        vertical_data.append(test_x[:, :, 12:49])
        vertical_data.append(test_x[:, :, 49:59])
        vertical_data.append(test_x[:, :, 59:76])
        for i in range(num_workers):
            V[:, :, i] = self.workers[i].forward(vertical_data[i])
        local_outputs = tf.convert_to_tensor(V, dtype=tf.float32)
        # print(local_outputs.shape)
        new_input = tf.reshape(local_outputs, [-1, num_units * num_workers])
        # with tf.device('/job:localhost/replica:0/task:0/device:GPU:0'):
        output = self.server.forward(new_input)
        loss = self.loss(test_y, output)
        testing_loss = self.test_cost(loss).numpy()
        testing_accuracy = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(test_y, output)).numpy()
        # test_loss.append(testing_loss)
        # test_accuracy.append(testing_accuracy)

        # compute total test loss and accuracy
        # total_loss = sum(test_loss) / len(test_loss)
        # total_accuracy = sum(test_accuracy) / len(test_accuracy)
        # print("Done")
        M = tf.math.confusion_matrix(test_y.numpy(), tf.math.round(output).numpy())
        # print(index)
        # print(M.numpy())
        auc = ROC_curve(test_y.numpy(), output.numpy(), name)
        # print(auc)
        return testing_loss, testing_accuracy, M, auc


if __name__ == "__main__":
    # s = compile1.system()
    # s.syn(async_train_dataset, test_dataset)
    s = system()
    s.pool.map(partial(s.asyn, train_dataset=train_dataset, test_dataset=test_dataset), list(range(num_workers)))
    # [0, 1, 2, 3])
