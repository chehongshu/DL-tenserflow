"""
@Author:Che_Hongshu
@Function: tools for CNN-CIFAR-10 dataset
@Modify:2018.3.5
@IDE: pycharm
@python :3.6
@os : win10
"""

import tensorflow as tf
"""
函数说明: 得到weights变量和weights的loss
Parameters:
   shape-维度
   stddev-方差
   w1-
Returns:
    var-维度为shape，方差为stddev变量
CSDN:
    http://blog.csdn.net/qq_33431368
Modify:
    2018-3-5
"""
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var
"""
函数说明: 得到总体的losses
Parameters:
   logits-通过神经网络之后的前向传播的结果
   labels-图片的标签
Returns:
   losses
CSDN:
    http://blog.csdn.net/qq_33431368
Modify:
    2018-3-5
"""
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
        (logits=logits, labels=labels, name='total_loss')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entorpy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

"""
函数说明: 对变量进行min max 和 stddev的tensorboard显示
Parameters:
    var-变量
    name-名字
Returns:
    None
CSDN:
    http://blog.csdn.net/qq_33431368
Modify:
    2018-3-5
"""
def variables_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'+name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var-mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)
        tf.summary.histogram()
