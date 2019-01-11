# -*- coding:UTF-8 -*-

import tensorflow as tf


t = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)
t.write(0, 1)
t.write(1, 33)
with tf.Session() as session:
    print session.run(t.read(1).eval())