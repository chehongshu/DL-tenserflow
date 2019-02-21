import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""
place_holder+conv2d+pool2d+circle+fcn(reshape)+out+get logits+ loss + optimizer+sess(train)+acc
"""
minst = input_data.read_data_sets('MINST_data', one_hot=True)

learning_rate = 0.001
batch_size = 128
num_step = 2000
display_step = 100

input_size = 784
num_labels = 10
drop_out = 0.9

xs = tf.placeholder(tf.float32, [batch_size, input_size])
ys = tf.placeholder(tf.float32, [batch_size, num_labels])
keep_prob = tf.placeholder(tf.float32)


def conv2d(input, W, b, stride=1):

    conv = tf.nn.conv2d(input, W, strides=[1, stride, stride, 1], padding='SAME')

    return tf.nn.bias_add(conv, b)

def maxpool2d(input, stride=2):
    return tf.nn.max_pool(input, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')


def inference(x, W, b, dropout):

    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = conv2d(x, W['wc1'], b['bc1'])
    pool1 = maxpool2d(conv1)
    conv2 = conv2d(pool1, W['wc2'], b['bc2'])
    pool2 = maxpool2d(conv2)
    print pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, shape=[batch_size, -1])
    dim = pool2.shape.as_list()
    Weight_fc = tf.Variable(tf.random_normal([int(dim[1]), 1024]))
    bia_fc = tf.Variable(tf.random_normal([1024]))
    fc = tf.nn.relu(tf.add(tf.matmul(pool2, Weight_fc), bia_fc))
    fc = tf.nn.dropout(fc, dropout)
    out = tf.add(tf.matmul(fc, W['out']), b['out'])

    return out

Weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'out': tf.Variable(tf.random_normal([1024, 10]))
}

Bias = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'out': tf.Variable(tf.random_normal([10]))
}


logits = inference(xs, Weights, Bias, keep_prob)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=ys)
loss = tf.reduce_mean(cross_entropy)

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

corect_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(logits, 1))
acc = tf.reduce_mean(tf.cast(corect_prediction, tf.float32))


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(num_step):
        x_input, y_input = minst.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={xs: x_input, ys: y_input, keep_prob: drop_out})
        if i % display_step or i == 1:
            ac, los = sess.run([acc, loss], feed_dict={xs: x_input, ys: y_input, keep_prob: 1})
            print("acc is "+"{:.3f}".format(ac)+"   loss is   "+"{:.3f}".format(los))
    print('over')
    test_ac = sess.run(acc, feed_dict={xs: minst.test.images[:128], ys: minst.test.labels[:128], keep_prob: 1})
    print("test acc is ", test_ac)