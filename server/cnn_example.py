# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:14:09 2019

@author: wangjingyi
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import time
mnist = input_data.read_data_sets(r'.\MNIST_data',one_hot=True)

#卷积层
class Convolution:
    def __init__(self):
        # 输出：[batch,14,14,16]
        self.filter1 = tf.Variable(tf.truncated_normal([3,3,1,16], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([16]))
        # 输出：[batch,7,7,64]
        self.filter2 = tf.Variable(tf.truncated_normal([3,3,16,32], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([32]))
        # 输出：[batch,4,4,64]
        self.filter3 = tf.Variable(tf.truncated_normal([3,3,32,64], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([64]))
        # 输出：[batch,2,2,64]
        self.filter4 = tf.Variable(tf.truncated_normal([2,2,64,64], stddev=0.1))
        self.b4 = tf.Variable(tf.zeros([64]))

    def forward(self,in_x):
        conv1 = tf.nn.leaky_relu(tf.add(tf.nn.conv2d(in_x,
                                                     self.filter1,
                                                     [1, 2, 2, 1],
                                                     padding='SAME'), self.b1))
        conv2 = tf.nn.leaky_relu(tf.add(tf.nn.conv2d(conv1,
                                                     self.filter2,
                                                     [1, 2, 2, 1],
                                                     padding='SAME'), self.b2))
        conv3 = tf.nn.leaky_relu(tf.add(tf.nn.conv2d(conv2,
                                                     self.filter3,
                                                     [1, 2, 2, 1],
                                                     padding='SAME'), self.b3))
        conv4 = tf.nn.leaky_relu(tf.add(tf.nn.conv2d(conv3,
                                                    self.filter4,
                                                    [1, 2, 2, 1],
                                                    padding="SAME"), self.b4))
        return conv4
#全连接层
class MLP:
    def __init__(self):
        self.in_w = tf.Variable(tf.truncated_normal([2*2*64, 100], stddev=0.1))
        self.in_b = tf.Variable(tf.truncated_normal([100]))

        self.out_w = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))
        self.out_b = tf.Variable(tf.zeros([10]))
    def forward(self,mlp_in_x):

        mlp_layer = tf.nn.leaky_relu(tf.add(tf.matmul(mlp_in_x, self.in_w), self.in_b))
        out_layer = tf.nn.leaky_relu(tf.add(tf.matmul(mlp_layer, self.out_w), self.out_b))

        return out_layer
    
#CNN神经网络类
class CNNnet:
    def __init__(self):
        #卷积层
        self.conv = Convolution()
        #全连接层
        self.mlp = MLP()

        self.in_x = tf.placeholder(dtype=tf.float32, shape=[None,28,28,1])
        self.in_y = tf.placeholder(dtype=tf.float32, shape=[None,10])
        #向前网络结构
        self.forward()
        #向后网络结构图
        self.backward()

    def forward(self):
        # (100, 2, 2, 64)
        self.conv_layer = self.conv.forward(self.in_x)
        mlp_in_x = tf.reshape(self.conv_layer,[-1,2*2*64])
        self.out_layer = self.mlp.forward(mlp_in_x)
    def backward(self):
        # pass
        self.loss = tf.reduce_mean((self.out_layer-self.in_y)**2)
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)
        
#train the model
#训练神经网络
#if __name__ == '__main__':
#    cnn = CNNnet()
#    with tf.Session() as sess:
#        init = tf.global_variables_initializer()
#        sess.run(init)
#        saver = tf.train.Saver()
#        loss_sum = []
#        time1 = time.time()
#        for epoch in range(10000):
#            xs,xy = mnist.train.next_batch(100)
#            loss,_ = sess.run([cnn.loss, cnn.opt], feed_dict={cnn.in_x: np.reshape(xs,[100,28,28,1]), cnn.in_y:xy})
#            if epoch% 200 == 0:
#                loss_sum.append(loss)
#                saver.save(sess, r'.\CNN_Model\CNNTrain1.ckpt')
#                test_xs,test_xy = mnist.test.next_batch(5)
#                out_layer = sess.run([cnn.out_layer], feed_dict={cnn.in_x: np.reshape(test_xs,[5,28,28,1])})
#                out_layer = np.array(out_layer).reshape((5,10))
#
#                out = np.array(out_layer).argmax(axis=1)
#                test_y = np.array(test_xy).argmax(axis=1)
#                accuracy = np.mean(out == test_y)
#                print('epoch:\t',epoch, 'loss:\t',loss,'accuracy:\t',accuracy,'原始数据：',test_y,"预测数据：",test_y)
#        time2 = time.time()
#        print('训练时间：\t',time2-time1)
#        plt.figure('CNN_Loss图')
#        plt.plot(loss_sum,label='Loss')
#        plt.legend()
#        plt.show()

#训练完成以后，使用模型进行测试，测试识别准确率
if __name__ == '__main__':
    cnn = CNNnet()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        accuracy_sum = []
        time1 = time.time()
        for epoch in range(1000):
            saver.restore(sess, r'.\CNN_Model\CNNTrain1.ckpt')
            test_xs,test_xy = mnist.test.next_batch(100)
            out_layer = sess.run([cnn.out_layer], feed_dict={cnn.in_x: np.reshape(test_xs,[100,28,28,1])})
            out_layer = np.array(out_layer).reshape((100,10))

            out = np.array(out_layer).argmax(axis=1)
            test_y = np.array(test_xy).argmax(axis=1)
            accuracy = np.mean(out == test_y)
            accuracy_sum.append(accuracy)
            print('epoch:\t',epoch, 'accuracy:\t',accuracy,)
    time2 = time.time()
    print('训练时间：\t',time2-time1)
    total_accuracy = sum(accuracy_sum)/len(accuracy_sum)
    print('总准确率：\t',total_accuracy)
    #  正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure('CNN_Accuracy图')
    plt.plot(accuracy_sum,'o',label='Accuracy')
    plt.title('Accuracy：{:.2f}%'.format(total_accuracy*100))
    plt.legend()
    plt.show()



