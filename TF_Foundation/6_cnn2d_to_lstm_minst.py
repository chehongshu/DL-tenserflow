import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

minst = input_data.read_data_sets('./MINST_data/', one_hot=True)

global_step = 30000
display_step = 100
lr = 0.001
batch_size = 128

num_inputs = 784
num_inputs2 = 32
num_labels = 10
time_step = 32
hidden_num = 512

kernel1 = 3
kernel2 = 3
w_out_channel1 = 64
w_out_channel2 = 128
fcn_layer_num1 = 1024
fcn_layer_num2 = 2048


xs = tf.placeholder(tf.float32, [batch_size, num_inputs])
ys = tf.placeholder(tf.float32, [batch_size, num_labels])

def con2d(x, W, b, strdie=2):

    conv = tf.nn.conv2d(x, filter=W, strides=[1, strdie, strdie, 1], padding='SAME')
    return tf.nn.relu(tf.nn.bias_add(conv, b))

def pool2d(x, stride=2):

    pool = tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')
    return pool

def fcn(x, W, b):

    fc = tf.add(tf.matmul(x, W), b)
    return fc

def lstm(x):

    x = tf.unstack(x, time_step, axis=1)
    cell = rnn.BasicLSTMCell(num_units=hidden_num, dtype=tf.float32)
    out_put, state = rnn.static_rnn(cell=cell, inputs=x, dtype=tf.float32)

    return out_put

def inference(x, W, b):
    x = tf.reshape(x, [-1, 28, 28, 1])

    conv1 = con2d(x, W['c1'], b['c1'], strdie=1)
    pool1 = pool2d(conv1)
    conv2 = con2d(pool1, W['c2'], b['c2'])
    pool2 = pool2d(conv2)

    pool2 = tf.reshape(pool2, [batch_size, -1])
    dim = pool2.get_shape().as_list()[1]
    W_fc = tf.Variable(tf.random_normal([dim, fcn_layer_num1]))
    b_fc = tf.Variable(tf.random_normal([fcn_layer_num1]))
    fc1 = fcn(pool2, W_fc, b_fc)

    x2 = tf.reshape(fc1, [-1, time_step, num_inputs2])
    out_put = lstm(x2)
    fc2 = fcn(out_put[-1], W['fc2'], b['fc2'])
    out = fcn(fc2, W['out'], b['out'])

    return out


W = {'c1': tf.Variable(tf.random_normal([kernel1, kernel1, 1, w_out_channel1])),
     'c2': tf.Variable(tf.random_normal([kernel2, kernel2, w_out_channel1, w_out_channel2])),
     'fc2':tf.Variable(tf.random_normal([hidden_num, fcn_layer_num2])),
     'out':tf.Variable(tf.random_normal([fcn_layer_num2, num_labels]))
}
b = {'c1': tf.Variable(tf.random_normal([w_out_channel1])),
     'c2': tf.Variable(tf.random_normal([w_out_channel2])),
     'fc2':tf.Variable(tf.random_normal([fcn_layer_num2])),
     'out':tf.Variable(tf.random_normal([num_labels]))
}
logits = inference(xs, W, b)

prediction = tf.nn.softmax(logits)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(ys, 1))

loss_op = tf.reduce_mean(cross_entropy)

train_op = tf.train.AdagradOptimizer(learning_rate=lr).minimize(loss_op)

correct_prediction = tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1)), tf.float32)

acc_op = tf.reduce_mean(correct_prediction)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(global_step):
        x_input, y_input = minst.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={xs: x_input,
                                      ys: y_input})
        if i% display_step == 0:
            x_test, y_test = minst.test.next_batch(batch_size)
            loss, acc = sess.run([loss_op, acc_op], feed_dict={xs: x_test,
                                      ys: y_test})
            print("step is "+str(i)+" loss is " + str(loss)+" , acc is "+str(acc))

    print('over')
    x_val, y_val = minst.validation.next_batch(batch_size)
    loss, acc = sess.run([loss_op, acc_op],feed_dict={xs: x_val, ys:y_val})
    print('for val, loss is '+str(loss)+" acc is "+str(acc))