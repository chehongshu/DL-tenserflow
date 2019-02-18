import tensorflow as tf
a = [9]
oh = tf.one_hot(a, depth=10)
with tf.Session() as session:
    one_hot_data = session.run(oh)
    print(one_hot_data)