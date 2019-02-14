import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.randint(-2, 2, 222).astype(np.float32)
y_data = x_data*2.0 + 1.0

with tf.name_scope('input_placeholder'):
    x_input = tf.placeholder(tf.float32, shape=(222))
    y_input = tf.placeholder(tf.float32, shape=(222))
with tf.name_scope('Variable'):
    weights = tf.Variable(tf.random_uniform([1], -1, 1))
    bias = tf.Variable(tf.zeros([1]))
with tf.name_scope('inference'):
    y = weights*x_data + bias
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y-y_data))
with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
plt.figure()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for step in range(500):
        sess.run(optimizer, feed_dict={x_input: x_data, y_input: y_data})
        if step % 20 == 0:
            w, b, los = sess.run([weights, bias, loss])
            y_result = w*x_data + b

        if step % 100 == 0:
            plt.plot(x_data, y_data)
            plt.plot(x_data, y_result)
            plt.xlabel("x")
            plt.xlabel("y")
            plt.xlim([-10, 10])
            plt.ylim([-10, 10])
            plt.show()



