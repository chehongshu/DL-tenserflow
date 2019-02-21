import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

display_step = 200
step = 10000
batchsize = 128
time_step = 28

lr = 0.0001
num_inputs = 28
num_labels = 10
hidden_num = 128


minst = input_data.read_data_sets('MINST_data', one_hot=True)

with tf.name_scope('input_holder'):
    xs = tf.placeholder(tf.float32, [None, time_step, num_inputs])
    ys = tf.placeholder(tf.float32, [None, num_labels])

def static_LSTM_inference(x):
    W = tf.Variable(tf.random_normal([hidden_num, num_labels]))
    b = tf.Variable(tf.random_normal([num_labels]))

    x = tf.unstack(x, time_step, axis=1)
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_num)
    out, state = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.add(tf.matmul(out[-1], W), b)
with tf.name_scope('inference'):
    logits = static_LSTM_inference(xs)

with tf.name_scope('loss_train'):

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(ys, 1)))

    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
with tf.name_scope('acc'):
    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(ys, 1)), tf.float32)

    accuracy = tf.reduce_mean(correct_prediction)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(step):
        x, y = minst.train.next_batch(batchsize)

        sess.run(train_op, feed_dict={xs: x.reshape([-1, time_step, num_inputs]), ys: y})
        if i == 1 or i % display_step == 0:
            x, y = minst.test.next_batch(batchsize)
            los, acc = sess.run([loss, accuracy], feed_dict={
                xs: x.reshape([-1, time_step, num_inputs]),
                ys: y})

            print(" loss is "+str(los)+"  acc is "+ str(acc))

