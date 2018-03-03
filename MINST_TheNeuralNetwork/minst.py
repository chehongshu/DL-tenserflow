"""
@Author:Che_Hongshu
@Modify:2018.2.17
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784  #输入层的节点数
OUTPUT_NODE = 10  #输出层的节点数

LAYER1_NODE = 500 #隐藏层节点数

BATCH_SIZE = 100 #一个训练batch中的训练数据个数
LEARNING_RATE_BASE = 0.8 #基础的学习率
LEARNING_RATE_DECAY = 0.99 #学习率的衰减率

REGULARIZATION_RATE = 0.0001 #正则化的系数
TRAINING_STEPS = 30000  #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率

"""
函数说明: 计算神经网络的前向传播的结果
Parameters:
    input_tensor-输入的张量
    avg_class-计算平均值的类-用于滑动平均模型
    weights1-隐藏层的权重参数
    biases1-隐藏层的偏置参数
    weights2-输出层的权重参数
    biases2-输出层的偏置参数
Returns:
    神经网络前向传播的结果
CSDN:
    http://blog.csdn.net/qq_33431368
Modify:
    2018-2-18

"""
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):

    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1)+biases1)
        return tf.matmul(layer1, weights2)+biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1))+
                            avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2))+avg_class.average(biases2)

"""
函数说明: 训练模型
Parameters:
    mnist-数据集
Returns:
    神经网络前向传播的结果
CSDN:
    http://blog.csdn.net/qq_33431368
Modify:
    2018-2-18

"""
def train(mnist):
    #定义读入输入输出数据的地方
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input') #定义输入的数据的地方
    y_= tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input') #定义输出数据的地方
    #定义隐藏层的参数， W1和b1
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))#正态分布的数据，方差为0.1
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE])) #常量0.1
    #定义输出层的参数   W2和b2
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE])) #常量0.1
    #前向传播的结果
    y = inference(x, None, weights1, biases1, weights2, biases2)
    #定义存储训练轮数的变量
    global_step = tf.Variable(0, trainable=False)
    #定义一个滑动平均的类
    variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    #定义一个更新变量滑动平均的操作 除了 trainable = False的变量
    variables_averages_op = variables_averages.apply(tf.trainable_variables())
    #计算使用滑动平均之后的前向传播结果
    average_y = inference(x, variables_averages, weights1, biases1, weights2, biases2)
    #计算交叉熵  sparse_softmax_cross_entropy_with_logits的引用需要明确 logits=  labels=
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    #计算当前batch的所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #正则化，防止过拟合， L2
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算正则化损失
    regularization = regularizer(weights1)+regularizer(weights2)
    #总损失=交叉熵损失+正则化损失
    loss = cross_entropy_mean + regularization
    #设置衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,#基础学习率
        global_step,#当前迭代的次数
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY# 学习率的衰减速度
    )
    #定义优化算法
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    #更新每个参数的滑动平均值
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    #tf.argmax(average_y,1) 返回的为每一行的最大值的索引
    #验证使用滑动平均模型的前向传播的结果是否正确 相同返回TURE 不相同返回False
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    #正确率  tf.cast 将 TURE和FALSE 转换成float32类型的 1.0  0.0
    #计算平均值就为这组数据的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #准备验证数据
        #一般神经网络的训练过程中会通过验证数据来大致判断停止的条件和评判训练的效果
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        #准备测试集。作为模型优劣的最后评价标准
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        #迭代的训练神经网络
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step,validation acc"
                      "using average model is %g"%(i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d steps,test accuracy"
              " using average model is %g"%(TRAINING_STEPS, test_acc))

def main(argv=None):
    MNIST =input_data.read_data_sets('./MINST_data', one_hot=True) # 读取数
    train(MNIST)

if __name__ == '__main__':
    tf.app.run()