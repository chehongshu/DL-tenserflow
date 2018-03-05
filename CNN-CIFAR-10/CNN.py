"""
@Author:Che_Hongshu
@Function: CNN-CIFAR-10 dataset
@Modify:2018.3.5
@IDE: pycharm
@python :3.6
@os : win10
"""
from tensorflow.models.tutorials.image.cifar10 import cifar10
from tensorflow.models.tutorials.image.cifar10 import cifar10_input

import tensorflow as tf
import numpy as np
import time
import tools
max_steps = 3000 # 训练轮数
batch_size = 128  #一个bacth的大小
data_dir = './cifar-10-batches-bin' #读取数据文件夹
LOG_DIR = './LOG'

#下载CIFAR数据集 如果不好用直接
# http://www.cs.toronto.edu/~kriz/cifar.html 下载CIFAR-10 binary version 文件解压放到相应的文件夹中
#cifar10.maybe_download_and_extract()
#得到训练集的images和labels
#print(images_train) 可知是一个shape= [128, 24, 24, 3]的tensor
images_train, labels_train = cifar10_input.\
    distorted_inputs(data_dir=data_dir, batch_size=batch_size)
#得到测试集的images和labels
images_test, labels_test = cifar10_input.\
    inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)
#以上两个为什么分别用distorted_inputs and inputs  请go to definition查询
#创建输入数据的placeholder
with tf.name_scope('input_holder'):
    image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
    label_holder = tf.placeholder(tf.int32, [batch_size])
#下面的卷积层的 weights的l2正则化不计算， 一般只在全连接层计算正则化loss
#第一个conv层
#5*5的卷积核大小，3个channel ，64个卷积核， weight的标准差为0.05
with tf.name_scope('conv1'):
    #加上更多的name_scope 使graph更加清晰好看，代码也更加清晰
    with tf.name_scope('weight1'): #权重
        weight1 = tools.variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
        #运用tensorboard进行显示
        tools.variables_summaries(weight1, 'conv1/weight1')
    kernel1 = tf.nn.conv2d(image_holder, weight1, strides=[1, 1, 1, 1], padding='SAME')
    with tf.name_scope('bias1'): #偏置
        bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
        tools.variables_summaries(bias1, 'conv1/bias1')
    with tf.name_scope('forward1'): #经过这个神经网络的前向传播的算法结果
        conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))#cnn加上bias需要调用bias_add不能直接+
#第一个最大池化层和LRN层
with tf.name_scope('pool_norm1'):
    with tf.name_scope('pool1'):
        # ksize和stride不同 ， 多样性
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 3, 3, 1], padding='SAME')
    with tf.name_scope('LRN1'):
        #LRN层可以使模型更加
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

#第二层conv层 input: 64   size = 5*5   64个卷积核
with tf.name_scope('conv2'):
    with tf.name_scope('weight2'):
        weight2 = tools.variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
        tools.variables_summaries(weight2, 'conv2/weight2')
    kernel2 = tf.nn.conv2d(norm1, weight2, strides=[1, 1, 1, 1], padding='SAME')
    with tf.name_scope('bias2'):
        bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
        tools.variables_summaries(bias2, 'conv2/bias2')
    with tf.name_scope('forward2'):
        conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))

#第二个LRN层和最大池化层
with tf.name_scope('norm_pool2'):
    with tf.name_scope('LRN2'):
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
    with tf.name_scope('pool2'):
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
# 全连接网络
with tf.name_scope('fnn1'):
    reshape = tf.reshape(pool2, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    with tf.name_scope('weight3'):
        weight3 = tools.variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
        tools.variables_summaries(weight3, 'fnn1/weight3')
    with tf.name_scope('bias3'):
        bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
        tools.variables_summaries(bias3, 'fnn1/bias3')
    local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

with tf.name_scope('fnn2'):
    with tf.name_scope('weight4'):
        weight4 = tools.variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
    with tf.name_scope('bias4'):
        bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
    local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)
with tf.name_scope('inference'):
    with tf.name_scope('weight5'):
        weight5 = tools.variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, w1=0.0)
    with tf.name_scope('bias5'):
        bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
    logits = tf.add(tf.matmul(local4, weight5), bias5)


with tf.name_scope('loss_func'):
    #求出全部的loss
    loss = tools.loss(logits, label_holder)
    tf.summary.scalar('loss', loss)

with tf.name_scope('train_step'):
    #调用优化方法Adam，这里学习率是直接设定的自行可以decay尝试一下
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=step)
    top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

#创建会话
sess = tf.InteractiveSession()
#变量初始化
tf.global_variables_initializer().run()
#合并全部的summary
merged = tf.summary.merge_all()
#将日志文件写入LOG_DIR中
train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
#因为数据集读取需要打开线程，这里打开线程
tf.train.start_queue_runners()
#开始迭代训练
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    summary, _, loss_value = sess.run([merged, train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    #每步进行记录
    train_writer.add_summary(summary, step)
    duration = time.time() - start_time
    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        #训练一个batch的time
        sec_per_batch = float(duration)
        format_str = ('step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

num_examples = 10000
import math
num_iter = int(math.ceil(num_examples/batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1
precision = true_count/total_sample_count

print('precision = %.3f' % precision)
