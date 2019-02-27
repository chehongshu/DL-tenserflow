import numpy as np
import tensorflow as tf

batchsize = 2

# define datset including features and labels
features = np.array([[1, 2, 3],
            [2, 3, 4],
            [4, 5, 6],
            [5, 6, 7]])

labels = np.array([0, 1, 2, 3])


def get_batch_data(images, label):
    input_queue = tf.train.slice_input_producer([images, label], shuffle=False, num_epochs=2)
    image_batch, label_batch = tf.train.shuffle_batch(input_queue, batch_size=batchsize,
                                                      capacity=64, min_after_dequeue=5)
    return image_batch, label_batch
# get onehot labels
label_one_hot = tf.one_hot(labels, depth=4)
# get shuffle dataset batch
images, label = get_batch_data(features, label_one_hot)

# run
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)
try:
    while not coord.should_stop():
        i, l = sess.run([images, label])
        print i
        print l
except tf.errors.OutOfRangeError:
    print('Done training')
finally:
    coord.request_stop()

coord.join(threads)
sess.close()