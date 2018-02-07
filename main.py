from tool import *
from model import *

import os

# Hyperparameters
flags = tf.app.flags
flags.DEFINE_string('dir_prefix', 'E:\\Dropbox\\Code\\FMNIST\\data\\', 'directory where MNIST is located')
flags.DEFINE_string('save_dir', '.\\output', 'directory where model graph and weights are saved')
flags.DEFINE_float('test_train_ratio', 0.85, 'train/raw_data ratio')
flags.DEFINE_float('vali_train_ratio', 0.9, 'vali/org_train ratio')
flags.DEFINE_float('REG_COEFF', 0.1, 'regularization coefficient')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 100, '')
flags.DEFINE_integer('patience', 2, '')
flags.DEFINE_integer('random_seed', 62, '')
FLAGS = flags.FLAGS

# Load data
images = np.load(FLAGS.dir_prefix + 'fmnist_train_data.npy')
images = np.reshape(images, [-1, 28, 28, 1])
labels = np.load(FLAGS.dir_prefix + 'fmnist_train_labels.npy')
labels = np.array([[(1.0 if i == j else 0.0) for j in range(10)] for i in labels], dtype='float32')

# Split data
train_images, train_labels, test_images, test_labels = \
    random_split_data(images, labels, FLAGS.test_train_ratio)
train_images, train_labels, vali_images, vali_labels = \
    random_split_data(train_images, train_labels, FLAGS.vali_train_ratio)
images = labels = None


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='data_placeholder')
    output = layers_bundle(x)
    y = tf.placeholder(tf.float32, [None, 10], name='label')
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = cross_entropy + FLAGS.REG_COEFF * sum(regularization_losses)

    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        path_prefix = saver.save(session, os.path.join(FLAGS.save_dir,
                                                       "fmnist_inference"), global_step=global_step_tensor)
