# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:13:24 2019

@author: wangjingyi
"""

import tensorflow as tf
from tensorflow.python.client import device_lib
import os
import time

# 设置tf记录的参数级别：
# 0 = 记录所有log信息
# 1 = 记录INFO级别信息
# 2 = INFO 和 WARNING 信息不记录
# 3 = INFO, WARNING 和 ERROR messages 信息不记录
# 如果在linux下，使用的语句是$ export TF_CPP_MIN_LOG_LEVEL=2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#################  获取当前设备所有GPU设备 ##################
def check_available_gpus():
    local_devices = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
    gpu_num = len(gpu_names)
    print('{0} GPUs are detected : {1}'.format(gpu_num, gpu_names))
    return gpu_num # 返回GPU个数

# 设置使用哪些GPU，实际的GPU12对我的程序来说就是GPU0
# 这里GPU需要是英伟达的GPU 支持CUDA计算
os.environ['CUDA_VISIBLE_DEVICES'] = '12, 13, 14, 15'
N_GPU = 4 # 定义GPU个数

# 定义网络中一些参数
BATCH_SIZE = 100*N_GPU
LEARNING_RATE = 0.001
EPOCHS_NUM = 1000
NUM_THREADS = 10

# 定义数据和模型路径
MODEL_SAVE_PATH = 'data/tmp/logs_and_models/'
MODEL_NAME = 'model.ckpt'
DATA_PATH = 'data/test_data.tfrecord'

#  Dataset的解析函数
def _parse_function(example_proto):
    dics = {
        'sample': tf.FixedLenFeature([5], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)}
    parsed_example = tf.parse_single_example(example_proto, dics)
    parsed_example['sample'] = tf.cast(parsed_example['sample'], tf.float32)
    parsed_example['label'] = tf.cast(parsed_example['label'], tf.float32)
    return parsed_example
# 读取数据并根据GPU个数进行均分
def _get_data(tfrecord_path = DATA_PATH, num_threads = NUM_THREADS, num_epochs = EPOCHS_NUM, batch_size = BATCH_SIZE, num_gpu = N_GPU):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    new_dataset = dataset.map(_parse_function, num_parallel_calls=num_threads)# 同时设置了多线程
# shuffle必须放在repeat前面，才能正确运行。否则会报错： Out of Range
    # shuffle打乱顺序
    shuffle_dataset = new_dataset.shuffle(buffer_size=10000)# shuffle打乱顺序
    # 定义重复训练多少次全部样本
    repeat_dataset = shuffle_dataset.repeat(num_epochs)
    batch_dataset = repeat_dataset.batch(batch_size=batch_size)
    iterator = batch_dataset.make_one_shot_iterator()# 创建迭代器
    next_element = iterator.get_next()
    x_split = tf.split(next_element['sample'], num_gpu)
    y_split = tf.split(next_element['label'], num_gpu)
    return x_split, y_split
# 由于对命名空间不理解，且模型的参数比较少，把参数的初始化放在外面，运行前只初始化一次。
# 但是，当模型参数多的时候，这样定义几百个会崩溃的。之后会详细介绍一下TF中共享变量的定义，解决此问题。
def _init_parameters():
    w1 = tf.get_variable('w1', shape=[5, 10], initializer=tf.random_normal_initializer(mean=0, stddev=1, seed=9))
    b1 = tf.get_variable('b1', shape=[10], initializer=tf.random_normal_initializer(mean=0, stddev=1, seed=1))
    w2 = tf.get_variable('w2', shape=[10, 1], initializer=tf.random_normal_initializer(mean=0, stddev=1, seed=0))
    b2 = tf.get_variable('b2', shape=[1], initializer=tf.random_normal_initializer(mean=0, stddev=1, seed=2))
    return w1, w2, b1, b2

# 计算平均梯度，平均梯度是对样本个数的平均
def average_gradients(tower_grads):
    avg_grads = []

    # grad_and_vars代表不同的参数（含全部gpu），如四个gpu上对应w1的所有梯度值
    for grad_and_vars in zip(*tower_grads):
        grads = []
        #循环不同gpu
        for g, _ in grad_and_vars:
            #扩展一个维度代表gpu，如w1=shape(5,10), 扩展后变为shape(1,5,10)
            expanded_g = tf.expand_dims(g, 0) 
            grads.append(expanded_g)
         # 在第一个维度上合并
        grad = tf.concat(grads, 0)
        # 计算平均梯度
        grad = tf.reduce_mean(grad, 0)
        
        # 变量参数
        v = grad_and_vars[0][1]
        # 将平均梯度和变量对应起来
        grad_and_var = (grad, v) 
        # 将不同变量的平均梯度append一起
        avg_grads.append(grad_and_var)
    # return 平均梯度
    return avg_grads

# 初始化变量
w1, w2, b1, b2 = _init_parameters()
# 获取训练样本
x_split, y_split = _get_data()
# 建立优化器
opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
tower_grads = []

# 将神经网络中前馈传输的图计算，分配给不同的gpu，训练不同的样本
for i in range(N_GPU):
    with tf.device("/gpu:%d" % i):
        y_hidden = tf.nn.relu(tf.matmul(x_split[i], w1) + b1)
        y_out = tf.matmul(y_hidden, w2) + b2
        y_out = tf.reshape(y_out, [-1])
        cur_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_out, labels=y_split[i], name=None)
        grads = opt.compute_gradients(cur_loss)
        tower_grads.append(grads)
    ######  建立一个session主要是想获取参数的具体数值，以查看是否对于每一个gpu来说都没有更新参数。
    ##### 在每个gpu上只是计算梯度，并没有更新参数。
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tower_grads)
        print('===============  parameter test sy =========')
        print(i)
        print(sess.run(b1))
        coord.request_stop()
        coord.join(threads)
# 计算平均梯度
grads = average_gradients(tower_grads)

# 用平均梯度更新模型参数
apply_gradient_op = opt.apply_gradients(grads)
# allow_soft_placement是当指定的设备如gpu不存在是，用可用的设备来处理。
# log_device_placement是记录哪些操作在哪个设备上完成的信息
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    #线程赤
    coord = tf.train.Coordinator()
    #设置多线程，进行分布式模拟测试
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #初始化参数
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    #开始分布式的训练
    for step in range(1000):
        start_time = time.time()
        sess.run(apply_gradient_op)

        duration = time.time() - start_time
        if step != 0 and step % 100 == 0:
            num_examples_per_step = BATCH_SIZE * N_GPU
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = duration / N_GPU
            print('step:', step, grads, examples_per_sec, sec_per_batch)
            print('=======================parameter b1============ :')
            print(sess.run(b1))
    coord.request_stop()
    coord.join(threads)