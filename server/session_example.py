# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:46:17 2019

@author: wangjingyi
"""

import tensorflow as tf
'''
wangjingyi tensorflow编程基础 session
'''

'''
1.编写hello world程序掩饰session的使用
   建立一个session,在session中输出hello TensorFlow    
'''

#定义一个常量
hello = tf.constant('hello TensorFlow')

#构造阶段完成后，才能启动图，启动图的第一步是创建一个Session对象，如果无任何创建函数，会话构造器将启动默认图
sess = tf.Session()
#通过session里面的run()函数来运行结果
print(sess.run(hello))
#或者
print(hello.eval(session=sess))
#任务完毕，关闭会话，Session对象在使用完毕后需要关闭以释放资源，除了显示调用close()外，也可以使用with代码块
sess.close()


'''
2. with session的使用
'''
a = tf.constant(3)
b = tf.constant(4)
with tf.Session() as sess:
    print(' a + b =  {0}'.format(sess.run(a+b)))
    print(' a * b =  {0}'.format(sess.run(a*b)))
    
    
'''
3.交互式session
'''
#进入一个交互式TensorFlow会话
sess = tf.InteractiveSession()

x = tf.Variable([1.0,2.0])
a = tf.constant([3.0,3.0])

#使用初始化器 initinalizer op的run()初始化x
x.initializer.run()

#增加一个减去sub op,从 x 减去 a，运行减去op，输出结果
sub = tf.subtract(x,a)
print(sub.eval())               #[-2. -1.]    


'''
4.注入机制
'''
a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)
add = a + b
product = a*b
with tf.Session() as sess:
    #启动图后，变量必须先经过'初始化' op 
    sess.run(tf.global_variables_initializer())
    print(' a + b =  {0}'.format(sess.run(add,feed_dict={a:3,b:4})))
    print(' a * b =  {0}'.format(sess.run(product,feed_dict={a:3,b:4})))
    #一次取出两个节点值
    print(' {0}'.format(sess.run([add,product],feed_dict={a:3,b:4})))
    
    
'''
5.指定GPU运算
'''
'''
设备用字符串进行标识. 目前支持的设备包括:
    "/cpu:0": 机器的 CPU.
    "/gpu:0": 机器的第一个 GPU, 如果有的话.
    "/gpu:1": 机器的第二个 GPU, 以此类推.
'''
with tf.Session() as sess:
    with tf.device("/cpu:0"):
        print(sess.run(product,feed_dict={a:3,b:4}))
        
'''
通过tf.ConfigProto来构建一个config，在config中指定相关的GPU，并且在session中传入参数config='自己创建的config'来指定GPU操作
tf.ConfigProto参数如下
log_device_placement = True:是否打印设备分配日志
allow_soft_placement = True：如果指定的设备不存在，允许TF自动分配设备
'''
config = tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
session = tf.Session(config = config)


'''
设置GPU使用资源
'''
#tf.ConfigProto生成之后，还可以按需分配GPU使用的资源
config.gpu_options.allow_growth = True
#或者
gpu_options = tf.GPUOptions(allow_growth = True)
config = tf.ConfigProto(gpu_options = gpu_options)

#给GPU分配固定大小得计算资源，如分配给tensorflow的GPU的显存的大小为：GPU显存x0.7
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.7)