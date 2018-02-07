import tensorflow as tf
import numpy as np
from functools import reduce


def i_regular_conv_layer(input_tensor,
                         num_filters,
                         activation_type,
                         kernel_regularization_type=None,
                         bias_regularization_type=None,
                         pool_strides=2,
                         pool_size=2,
                         kernel_size=(3, 3),
                         padding='same',
                         name='conv_block'):
    with tf.name_scope(name) as scope:
        conv_layers = [input_tensor, ]
        for i in num_filters:
            conv_layers.append(tf.layers.Conv2D(i, kernel_size, 1,
                                                padding=padding,
                                                activation=activation_type,
                                                kernel_regularizer=kernel_regularization_type,
                                                bias_regularizer=bias_regularization_type))
        pool = tf.layers.MaxPooling2D(pool_size, pool_strides, padding=padding)
        conv_layers_group = reduce(lambda lhs, rhs: rhs(lhs), conv_layers)
        output_tensor = pool(conv_layers_group)

        block_parameter_num = sum(map(lambda layer: layer.count_params(), conv_layers[1:]))+pool.count_params()
        print('Number of parameters in normal conv block: ', block_parameter_num)
        return output_tensor


def layers_bundle(input_tensor):
    with tf.name_scope('yt_model') as scope:
        conv_layers = i_regular_conv_layer(input_tensor, [64, 128], tf.nn.relu, pool_size=2, pool_strides=2, name='yt_core_model')
        flat = tf.reshape(conv_layers, [-1, 14*14*128])
        output = tf.layers.dense(flat, 10, name='output')
        tf.identity(output, name='model_output')
        return output
