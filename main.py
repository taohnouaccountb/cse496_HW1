import tensorflow as tf
from util import *
from model import *
import matplotlib.pyplot as plt
import os

flags = tf.app.flags
flags.DEFINE_string('dir_prefix', 'E:\\Dropbox\\Code\\FMNIST-Git\\data\\', 'directory where MNIST is located')
flags.DEFINE_string('save_dir', '.\\output', 'directory where model graph and weights are saved')
flags.DEFINE_float('test_train_ratio', 0.95, 'train/raw_data ratio')
flags.DEFINE_float('vali_train_ratio', 0.9, 'vali/org_train ratio')
flags.DEFINE_float('REG_COEFF', 0.001, 'regularization coefficient')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 100, '')
flags.DEFINE_integer('patience', 4, '')
flags.DEFINE_integer('random_seed', 62, '')
FLAGS = flags.FLAGS

# Hyperparameters

# Load data
images = np.load(FLAGS.dir_prefix + 'fmnist_train_data.npy')
images = np.reshape(images, [-1, 784])
labels = np.load(FLAGS.dir_prefix + 'fmnist_train_labels.npy')
labels = np.array([[(1.0 if i == j else 0.0) for j in range(10)] for i in labels], dtype='float32')

# Split data
train_images, train_labels, test_images, test_labels = \
    random_split_data(images, labels, FLAGS.test_train_ratio)
train_images, train_labels, vali_images, vali_labels = \
    random_split_data(train_images, train_labels, FLAGS.vali_train_ratio)
images = labels = None

if __name__ == '__main__':
    imgplot = plt.imshow(test_images[0].reshape((28, 28)))
    plt.show()

    x = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    x_normalized = x / 255.0
    output = layers_bundle(x_normalized, name='output')
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

        validation_val = []
        validation_count = 0
        batch_size = FLAGS.batch_size
        mean_total_loss = tf.reduce_mean(total_loss)

        train_num_examples = train_images.shape[0]
        test_num_examples = test_images.shape[0]
        vali_num_examples = vali_images.shape[0]
        early_pos = []
        for epoch in range(FLAGS.max_epoch_num):
            print('')
            print('Epoch: ' + str(epoch))

            # run gradient steps and report mean loss on train data
            ce_vals = []
            for i in range(train_num_examples // batch_size):
                batch_xs = train_images[i * batch_size:(i + 1) * batch_size, :]
                batch_ys = train_labels[i * batch_size:(i + 1) * batch_size, :]
                _, train_ce = session.run([train_op, mean_total_loss], {x: batch_xs, y: batch_ys})
                ce_vals.append(train_ce)
            avg_train_ce = sum(ce_vals) / len(ce_vals)
            print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))

            # report mean test loss
            ce_vals = []
            conf_mxs = []
            for i in range(test_num_examples // batch_size):
                batch_xs = test_images[i * batch_size:(i + 1) * batch_size, :]
                batch_ys = test_labels[i * batch_size:(i + 1) * batch_size, :]
                test_ce, conf_matrix = session.run([mean_total_loss, confusion_matrix_op], {x: batch_xs, y: batch_ys})

                ce_vals.append(test_ce)
                conf_mxs.append(conf_matrix)
            avg_test_ce = sum(ce_vals) / len(ce_vals)
            print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
            print('TEST CONFUSION MATRIX:')
            print(str(sum(conf_mxs)))

            # report validation loss
            ce_vals = []
            conf_mxs = []
            for i in range(vali_num_examples // batch_size):
                batch_xs = vali_images[i * batch_size:(i + 1) * batch_size, :]
                batch_ys = vali_labels[i * batch_size:(i + 1) * batch_size, :]
                validation_ce, vali_conf_matrix = session.run([mean_total_loss, confusion_matrix_op],
                                                              {x: batch_xs, y: batch_ys})
                ce_vals.append(validation_ce)
                conf_mxs.append(vali_conf_matrix)
            conf_mxs = sum(conf_mxs)
            count_wrong = 0
            count_sum = 0
            for i in range(conf_mxs.shape[0]):
                for j in range(conf_mxs.shape[1]):
                    count_sum += conf_mxs[i][j]
                    if i != j:
                        count_wrong += conf_mxs[i][j]
            vali_ratio = 1 - count_wrong * 1.0 / count_sum

            print("VALIDATION CORRECT RATIO: ", vali_ratio)
            avg_test_ce = sum(ce_vals) / len(ce_vals)
            validation_val.append(avg_test_ce)
            print('VALIDATION CROSS ENTROPY: ' + str(avg_test_ce))
            if len(validation_val) >= 2 and validation_val[-1] > validation_val[-2]:
                validation_count = validation_count + 1
                print("EARLY! ", validation_count)
            else:
                validation_count = 0
            if validation_count >= FLAGS.patience:
                early_pos.append(epoch)
                print('Early Stop!!')
                break
            if vali_ratio>0.88:
                file_name = "homework_1-0_{ep}_{rate}_{rate2}".format(ep=epoch,
                                                                      rate=int(vali_ratio * 100),
                                                                      rate2=int(vali_ratio * 10000 % 100))
                path_prefix = saver.save(session, os.path.join(FLAGS.save_dir, file_name), global_step=global_step_tensor)
                print(path_prefix)

        x = range(len(validation_val))
        y = validation_val
        plt.plot(x, y, '-', label='validation_loss')
        plt.legend()
        print('EARLY_POS')
        print(early_pos)
        vali_mat = [[i, validation_val[i]] for i in range(len(validation_val))]
        print('VALI_VAL_MATRIX:')
        for i in vali_mat:
            print(i)
