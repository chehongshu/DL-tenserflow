# -*- coding:UTF-8 -*-
"""
@Author:Che_Hongshu
@Modify:2018.12.28
@CSDN:http://blog.csdn.net/qq_33431368

"""

import numpy as np
import tensorflow as tf

TRAIN_DATA = "ptb.train"  # 训练数据
EVAL_DATA = "ptb.valid"   # 验证数据
TEST_DATA = "ptb.test"    #  测试数据
HIDDEN_SIZE = 300         #　隐藏层

NUM_LAYERS = 2            #　LSTM 层数
VOCAB_SIZE = 10000        #　词典规模（只要这么大的规模的特征词的数字表示）
TRAIN_BATCH_SIZE = 20     #　训练数据的batchsize
TRAIN_NUM_STEP = 35       #　训练数据截断长度

EVAL_BATCH_SIZE = 1       #　验证数据的batchsize
EVAL_NUM_STEP = 1        #　验证数据截断长度
NUM_EPOCH = 5            #　训练数据的轮数
LSTM_KEEP_PROB = 0.9      #　LSTM节点不被dropout的概率
EMBEDDING_KEEP_PROB = 0.9  #　词向量不被dropout的概率
MAX_GRAB_NORM = 5         #　用于控制梯度膨胀大小的上限
SHARE_EMB_AND_SOFTMAX = True  #　Softmax预词向量层之间共享参数

"""
function: class of LSTM
Parameters:
Returns:
CSDN:
    http://blog.csdn.net/qq_33431368
"""
class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        # 记录使用的 batch 大小和截断长度
        self.batch_size = batch_size
        self.num_steps = num_steps

        # 定义每一步的输入和预期输出, 两者的维度都是[ batch_size ,num_steps ]
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 定义使用 LSTM 结构为循环体结构且使用 dropout 的深层循环神经网络。
        dropout_keep_prob = LSTM_KEEP_PROB if is_training else 1.0  #训练时采用dropout

        #定义lstm
        lstm_cells = [

            #运用Dropout的LSTM，不同时刻不dropout，同一时刻dropout
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
                output_keep_prob=dropout_keep_prob
            )
            # NUM_LAYERS为层数,也就是LSTM为几层
            for _ in range(NUM_LAYERS)
        ]
        #创建多层深度lstm
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
        # 对创建的LSTM进行初始化
        #初始化最初的状态, 即全０的向量。这个量只在每个epoch初始化第一个batch才使用。
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # 定义单词的词向量矩阵。
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])
        # 将输入单词转化为词向量
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        #只在训练时进行dropout,测试和验证都不要dropout操作
        if is_training:
            inputs = tf.nn.dropout(inputs, EMBEDDING_KEEP_PROB)
        # 定义输出列表。在这里先将不同时刻 LSTM 结构的输出收集起来 , 再一起提供给softmax层
        outputs = []
        # lstm状态值
        state = self.initial_state
        with tf.variable_scope("RNN"):
            #numsteps为每次截断的序列长度
            for time_step in range(num_steps):
                #在第一个时刻声明ＬＳＴＭ使用的变量，在之后的时刻都需要复用之前定义好的变量
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                #　得到这个时刻lstm输出以及当前状态
                cell_output, state = cell(inputs[:, time_step, :], state)
                #　这个时间段的结构，因为一步一步来append，所以是相当于[[],[],[],[]]需要下面一步concat（，１）来使所有的结果在一个维度上
                outputs.append(cell_output)
        #　把输出进行调整维度(,HEDDEN_SIZE)
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
        # 是否共享参数（softmax层＋embedding层）\

        #Softmax摆: 将RNN在每个位置上的输出转化为各个单词的logits，也就是最后得出的每个单词是最终预测结果的概率。
        if SHARE_EMB_AND_SOFTMAX:
            weight = tf.transpose(embedding)
        else:
            weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        #最后经过全连接层输出的结果
        logits = tf.matmul(output, weight) + bias
        #交叉熵损失函数，算loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.targets, [-1]), logits=logits)
        #求出每个batch平均loss
        self.cost = tf.reduce_sum(loss)/batch_size
        #最终的state
        self.final_state = state

        if not is_training:
            return

        # 控制梯度大小,定义优化方法和训练步骤。
        trainable_variables = tf.trainable_variables()
        # 算出每个需要更新的值的梯度，并对其进行控制
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAB_NORM)
        # 利用梯度下降优化算法进行优化.学习率为1.0
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        #相当于minimize的第二步，正常来讲所得到的list[grads,vars]由compute_gradients得到，返回的是执行对应变量的更新梯度操作的op
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

"""
function: 使用给定的模型 model 在datasets　上运行 train op 并返回在全部数据上的 perplexity　值
Parameters:
    session-会话
    model-模型
    batches-批量值
    train_op-执行对应变量的更新梯度操作op
    output_log-
    step-训练步数
Returns:
     return step, np.exp(total_costs / iters)-步数和对应求出的perplexity
CSDN:
    http://blog.csdn.net/qq_33431368
"""
def run_epoch(session, model, batches, train_op, output_log, step):
    # 计算平均 perplexity 的辅助变量
    total_costs = 0.0
    iters = 0
    #得到final_state
    state = session.run(model.initial_state)
    # 训练一个 epoch
    for x, y in batches:
        # 在当前batch 上运行 train op 并计算损失值,　交叉炳损失函数计算的就是下一个单词为给定单词的概率。
        cost, state, _ = session.run([model.cost, model.final_state, train_op],
                                     {model.input_data: x, model.targets: y, model.final_state: state})
    # 总的loss
    total_costs += cost
    # 总的截断长度
    iters += model.num_steps

    if output_log and step % 100 == 0:
        print "After %d steps, perplexity is %.3f" % (step, np.exp(total_costs/iters))
    #训练次数
    step += 1

    return step, np.exp(total_costs / iters)

"""
function: 计算神经网络的前向传播的结果
Parameters:
    file_path-文件路径(文件已经是前面处理好的id文件了)
Returns:
    idlist-对于输入数据产生对应的转换为int的list
CSDN:
    http://blog.csdn.net/qq_33431368
"""
def read_data(file_path):
    with open(file_path, 'r') as fin: #打开文件
        id_string = " ".join([line.strip() for line in fin.readlines()]) #每一行读取，并用空格相连
    id_list = [int(w) for w in id_string.split()] #转换成id list
    return id_list

"""
function: 数据batching,产生最后输入数据格式
Parameters:
    id_list-文件的对应id文件，由read_data产生
    batch_size-batch的大小
    num_step-截断序列数据的长度
Returns:
    list(zip(data_batches, label_batches))-data,label的数据list
CSDN:
    http://blog.csdn.net/qq_33431368
"""
def make_batches(id_list, batch_size, num_step):
    # 计算总的 batch 数量。每个 batch 包含的单词数量是 batch_size＊num_step
    num_batches = (len(id_list) - 1) // (batch_size*num_step)
    # 从头开始取正好num_batches*batch_size*num_step
    data = np.array(id_list[: num_batches*batch_size*num_step])
    # 将数据整理成一个维度为[ batch_size, num_batches*numstep ]
    data = np.reshape(data, [batch_size, num_batches*num_step])
    # 相当于在第二维数据上竖着截取一部分数据
    data_batches = np.split(data, num_batches, axis=1)
    # 因为相当于一个时刻去预测下一个时刻，所以进行相应的+1，相当于每个时刻的预测真值都在下一时刻。
    label = np.array(id_list[1:num_batches*batch_size*num_step +1])
    label = np.reshape(label, [batch_size, num_batches*num_step])
    label_batches = np.split(label, num_batches, axis=1)

    return list(zip(data_batches, label_batches))


def main():
    # 定义初始化函数。
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    #      initializer: default initializer for variables within this scope.
    # tf.variable_scope(,initializer=initializer)相当于在这个scope中都是这样的初始化变量情况
    # #定义训练用的循环神经网络模型。
    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
    # 定义测试用的循环神经网络模型。它与 train model 共用参数 ,　但是测试使用全部的参数，所以没有dropout 。
    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)
    #train
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        # 生成train,test,eval的batches
        train_batches = make_batches(
            read_data(TRAIN_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_STEP
        )
        eval_batches = make_batches(
            read_data(EVAL_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP
        )
        test_batches = make_batches(
            read_data(TEST_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP
        )
        step = 0
        #进行NUM_EPOCH次迭代
        for i in range(NUM_EPOCH):
            print "In iteration: %d" % (i+1)
            step, train_pplx = run_epoch(session, train_model, train_batches, train_model.train_op, True, step)

            print "Epoch : %d train Perplexity: %.3f" % (i + 1, train_pplx)

            _, eval_pplx = run_epoch(session, eval_model, eval_batches, tf.no_op(), False, 0)

            print "Epoch: %d Eval Perplexity: %.3f" % (i + 1, eval_pplx)

        _, test_pplx = run_epoch(session, eval_model, test_batches, tf.no_op(), False, 0)
        print "Test Preplex: %.3f" % test_pplx


if __name__ == '__main__':
    main()

