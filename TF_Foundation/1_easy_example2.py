import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# create dataset
x_data = np.linspace(-1, 1, 600)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) -1.0 + noise

#create placeholder tensor
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

def add_layer(x, input_size, output_size, activation_func=None):
    w = tf.Variable(tf.random_normal([input_size, output_size]))
    b = tf.Variable(tf.zeros([output_size])+0.1)
    out = tf.matmul(x, w) + b
    if activation_func is None:
        out = out
    else:
        out = activation_func(out)
    return out


layer1 = add_layer(xs, 1, 20, activation_func=tf.nn.relu)

prediction = add_layer(layer1, 20, 1)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.scatter(x_data, y_data)
plt.ion()
plt.show()


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(2000):
        _, los = sess.run([train_op, loss], feed_dict={xs: x_data, ys: y_data})

        if i % 100:
            data_pre = sess.run(prediction, feed_dict={xs: x_data})
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            lines = ax.plot(x_data, data_pre, 'r-', lw=5)
            plt.pause(0.1)

