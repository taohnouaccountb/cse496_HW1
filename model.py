import tensorflow as tf
import numpy as np
from functools import reduce


def i_regular_dense_layers(input_tensor,
                          layers_meta,
                          name_scope='layers_block'):
    with tf.name_scope(name_scope) as scope:
        layers = [input_tensor, ]
        for i in layers_meta:
            cur_layer = tf.layers.Dense(i['size'],
                                        kernel_regularizer=i['k_reg'],
                                        bias_regularizer=i['b_reg'],
                                        activation=i['func'],
                                        name=i['name'])
            layers.append(cur_layer)
        layers_group = reduce(lambda lhs, rhs: rhs(lhs), layers)
        return layers_group


def layers_bundle(input_tensor):
    l2 = tf.contrib.layers.l2_regularizer
    relu = tf.nn.relu
    with tf.name_scope('v1_model') as scope:
        normal_layers = i_regular_dense_layers(input_tensor, [
            {'size': 512, 'k_reg': l2(scale=1.0), 'b_reg': l2(scale=1.0), 'func': relu, 'name': 'hidden1_layer'},
            {'size': 128, 'k_reg': l2(scale=1.0), 'b_reg': l2(scale=1.0), 'func': relu, 'name': 'hidden2_layer'},
            {'size': 10, 'k_reg': l2(scale=1.0), 'b_reg': l2(scale=1.0), 'func': None, 'name': 'output_layer'}
        ], name_scope='core_model')
        tf.identity(normal_layers, name='model_output')
        return normal_layers


'''
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
'''
