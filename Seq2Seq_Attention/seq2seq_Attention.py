# -*- coding:UTF-8 -*-
"""
@Function:带有Attention的seq2seq模型的训练文件
@Author:Che_Hongshu
@Modify:2018.1.9
"""
import tensorflow as tf

SRC_TRAIN_DATA = 'train.en'  # 源语言输入文件
TRG_TRAIN_DATA = 'train.zh'  # 目标语言输入文件
CHECKPOINT_PATH = './model/seq2seq_ckpt'  # checkpoint保存路径
HIDDEN_SIZE = 1024                  # LSTM的隐藏层规模
NUM_LAYERS = 2                      # 深层循环神经网络中LSTM结构的层数
SRC_VOCAB_SIZE = 10000              # 源语言词汇表大小
TRG_VOCAB_SIZE = 4000               # 目标语言词汇表大小
BATCH_SIZE = 100                    # 训练数据batch的大小
NUM_EPOCH = 5                       # 使用训练数据的轮数
KEEP_PROB = 0.8                     # 节点不被dropout的概率
MAX_GRAD_NORM = 5                   # 用于控制梯度膨胀的梯度大小上限
SHARE_EMB_AND_SOFTMAX = True        # 在softmax层和词向量层之间共享参数
MAX_LEN = 50                        # 限定句子的最大单词数量
SOS_ID = 1                          # 目标语言词汇表中<sos>的ID


"""
function: 数据batching,产生最后输入数据格式
Parameters:
    file_path-数据路径
Returns:
    dataset-　每个句子－对应的长度组成的TextLineDataset类的数据集对应的张量
CSDN:
    http://blog.csdn.net/qq_33431368
"""
def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)

    # map(function, sequence[, sequence, ...]) -> list
    # 通过定义可以看到，这个函数的第一个参数是一个函数，剩下的参数是一个或多个序列，返回值是一个集合。
    # function可以理解为是一个一对一或多对一函数，map的作用是以参数序列中的每一个元素调用function函数，返回包含每次function函数返回值的list。
    # lambda argument_list: expression
    # 其中lambda是Python预留的关键字,argument_list和expression由用户自定义
    # argument_list参数列表, expression 为函数表达式
    # 根据空格将单词编号切分开并放入一个一维向量
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # 将字符串形式的单词编号转化为整数
    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
    # 统计每个句子的单词数量，并与句子内容一起放入Dataset
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset

"""
function: 从源语言文件src_path和目标语言文件trg_path中分别读取数据，并进行填充和batching操作
Parameters:
    src_path-源语言，即被翻译的语言,英语.
    trg_path-目标语言，翻译之后的语言,汉语.
    batch_size-batch的大小
Returns:
    dataset-　每个句子－对应的长度　组成的TextLineDataset类的数据集
CSDN:
    http://blog.csdn.net/qq_33431368
"""
def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    # 首先分别读取源语言数据和目标语言数据
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)
    # 通过zip操作将两个Dataset合并为一个Dataset，现在每个Dataset中每一项数据ds由4个张量组成
    # ds[0][0]是源句子
    # ds[0][1]是源句子长度
    # ds[1][0]是目标句子
    # ds[1][1]是目标句子长度
    #https://blog.csdn.net/qq_32458499/article/details/78856530这篇博客看一下可以细致了解一下Dataset这个库，以及.map和.zip的用法
    dataset = tf.data.Dataset.zip((src_data, trg_data))

    # 删除内容为空(只包含<eos>)的句子和长度过长的句子
    def FilterLength(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        # tf.logical_and 相当于集合中的and做法，后面两个都为true最终结果才会为true，否则为false
        # tf.greater Returns the truth value of (x > y),所以以下所说的是句子长度必须得大于一也就是不能为空的句子
        # tf.less_equal Returns the truth value of (x <= y),所以所说的是长度要小于最长长度
        src_len_ok = tf.logical_and(tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
        return tf.logical_and(src_len_ok, trg_len_ok) #两个都满足才返回true

    # filter接收一个函数Func并将该函数作用于dataset的每个元素，根据返回值True或False保留或丢弃该元素，True保留该元素，False丢弃该元素
    # 最后得到的就是去掉空句子和过长的句子的数据集
    dataset = dataset.filter(FilterLength)

    # 解码器需要两种格式的目标句子：
    # 1.解码器的输入(trg_input), 形式如同'<sos> X Y Z'
    # 2.解码器的目标输出(trg_label), 形式如同'X Y Z <eos>'
    # 上面从文件中读到的目标句子是'X Y Z <eos>'的形式，我们需要从中生成'<sos> X Y Z'形式并加入到Dataset
    # 编码器只有输入,没有输出,而解码器有输入也有输出，输入为<sos>＋(除去最后一位eos的label列表)
    # 例如train.en最后都为2,ｉｄ为２就是eos
    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        # tf.concat用法 https://blog.csdn.net/qq_33431368/article/details/79429295
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))
    dataset = dataset.map(MakeTrgInput)

    # 随机打乱训练数据
    dataset = dataset.shuffle(10000)

    # 规定填充后的输出的数据维度
    padded_shapes = (
        (tf.TensorShape([None]),    # 源句子是长度未知的向量
         tf.TensorShape([])),       # 源句子长度是单个数字
        (tf.TensorShape([None]),    # 目标句子(解码器输入)是长度未知的向量
         tf.TensorShape([None]),    # 目标句子(解码器目标输出)是长度未知的向量
         tf.TensorShape([]))        # 目标句子长度(输出)是单个数字
    )
    # 调用padded_batch方法进行padding 和　batching操作
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)

    return batched_dataset

"""
function: seq2seq模型－Attention机制
Parameters:
Returns:
CSDN:
    http://blog.csdn.net/qq_33431368
"""
# attention 编码器双向循环，解码器单向循环
class NMTModel_Attention(object):
    # 在模型的初始化函数中定义模型要用到的变量
    def __init__(self):
        # 定义编码器和解码器所使用的LSTM结构
        self.enc_cell_fw = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)  #前向
        self.enc_cell_bw = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)  #反向

        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
        # 为源语言和目标语言分别定义词向量
        self.src_embedding = tf.get_variable('src_emb', [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable('trg_emb', [TRG_VOCAB_SIZE, HIDDEN_SIZE])
        # 定义softmax层的变量
        # 只有解码器需要用到softmax
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable('weight', [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable('softmax_loss', [TRG_VOCAB_SIZE])

    """
      function: 在forward函数中定义模型的前向计算图
      Parameters:
      　　MakeSrcTrgDataset函数产生的五种张量如下（全部为张量）
          src_input: 编码器输入（源数据）
          src_size : 输入大小
          trg_input：解码器输入（目标数据）
          trg_label：解码器输出（目标数据）
          trg_size：　输出大小
      Returns:
      CSDN:
          http://blog.csdn.net/qq_33431368
      """
    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]
        # 将输入和输出单词转为词向量
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)
        # 在词向量上进行dropout
        src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)
        # 编码器
        with tf.variable_scope('encoder'):
            # 构造编码器时，使用birdirectional_dynamic_rnn构造双向循环网络。
            # 双向循环网络的顶层输出enc_outputs是一个包含两个张量的tuple，每个张量的
            # 维度都是[batch_size, max_time, HIDDEN_SIZE],代表两个LSTM在每一步的输出
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(self.enc_cell_fw, self.enc_cell_bw, src_emb,
                                                                     src_size, dtype=tf.float32)
            # 将两个LSTM输出拼接为一个张量
            enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)

        # 使用dynamic_rnn构造解码器
        with tf.variable_scope('decoder'):
            # 选择注意力权重的计算模型。BahdanauAttention是使用一个隐藏层的前馈神经网络
            # memory_sequence_length是一个维度为[batch_size]的张量，代表batch中每个句子的长度
            # Attention需要根据这个信息把填充位置的注意里权重设置为0
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(HIDDEN_SIZE, enc_outputs,
                                                                       memory_sequence_length=src_size)
            # 将解码器的循环神经网络self.dec_cell和注意力一起封装成更高层的循环神经网络
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell, attention_mechanism,
                                                                 attention_layer_size=HIDDEN_SIZE)
            # 使用attention_cell和dynamic_rnn构造编码器
            # 这里没有指定init_state,也就是没有使用编码器的输出来初始化输入，而完全依赖注意力作为信息来源
            dec_outputs, _ = tf.nn.dynamic_rnn(attention_cell, trg_emb, trg_size, dtype=tf.float32)

        # 计算解码器每一步的log perplexity
        output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label, [-1]), logits=logits)
        # 在计算平均损失时，需要将填充位置的权重设置为0，以避免无效位置的预测干扰模型的训练
        label_weights = tf.sequence_mask(trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)
        # 定义反向传播操作
        trainable_variables = tf.trainable_variables()
        # 控制梯度大小，定义优化方法和训练步骤
        grads = tf.gradients(cost / tf.to_float(batch_size), trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
        return cost_per_token, train_op

"""
function: 使用给定的模型model上训练一个epoch，并返回全局步数，每训练200步便保存一个checkpoint
Parameters:
    session :  会议
    cost_op :  计算loss的操作op
    train_op：　训练的操作op
    saver：　　保存model的类
    step：　　　训练步数
Returns:
CSDN:
    http://blog.csdn.net/qq_33431368
"""
def run_epoch(session, cost_op, train_op, saver, step):
    # 训练一个epoch
    # 重复训练步骤直至遍历完Dataset中所有数据
    while True:
        try:
            # 运行train_op并计算cost_op的结果也就是损失值，训练数据在main()函数中以Dataset方式提供
            cost, _ = session.run([cost_op, train_op])
            # 步数为１０的倍数进行打印
            if step % 10 == 0:
                print('After %d steps, per token cost is %.3f' % (step, cost))
            # 每200步保存一个checkpoint
            if step % 200 == 0:
                saver.save(session, CHECKPOINT_PATH, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step

"""
function: 主函数
Parameters:
Returns:
CSDN:
    http://blog.csdn.net/qq_33431368
"""
def main():
    # 定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    # 定义训练用的循环神经网络模型
    with tf.variable_scope('nmt_model', reuse=None, initializer=initializer):
        train_model = NMTModel_Attention()
    # 定义输入数据
    data = MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()
    # 定义前向计算图，输入数据以张量形式提供给forward函数
    cost_op, train_op = train_model.forward(src, src_size, trg_input, trg_label, trg_size)
    # 训练模型
    # 保存模型
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        # 初始化全部变量
        tf.global_variables_initializer().run()
        # 进行NUM_EPOCH轮数
        for i in range(NUM_EPOCH):
            print('In iteration: %d' % (i + 1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)


if __name__ == '__main__':
    main()