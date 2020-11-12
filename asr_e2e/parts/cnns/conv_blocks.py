from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

from six.moves import range

import tensorflow as tf

layers_dict = {
        "conv1d": tf.layers.conv1d,
        "sep_conv1d": tf.layers.separable_conv1d,
        "conv2d": tf.layers.conv2d}

def conv_bn_actv(layer_type, name, inputs, filters,
                 kernel_size, activation_fn, strides,
                 padding, regularizer, training,
                 data_format, bn_momentus, bn_epsilon,
                 dilation = 1):

    layer = layers_dict[layer_type]
    if layer_type == "seq_conv1d":
        conv = layer(
                name = "{}".format(name),
                inputs = inputs,
                filters = filters,
                kernel_size = kernel_size,
                strides = strides,
                padding = padding,
                dilation_rate = dilation,
                depthwise_regularizer = regularizer,
                pointwise_regularizer = regularizer,
                use_bias = False,
                data_format = data_format)

    else:
        conv = layer(
                name = "{}".format(name),
                inputs = inputs,
                filters = filters,
                kernel_size = kernel_size,
                strides = strides,
                padding = padding,
                dilation_rate = dilation,
                kernel_regularizer = regularizer,
                use_bias = False,
                data_format = data_format)

    squeeze = False
    if "conv1d" in layer_type:
        axis = 1 if data_format == "channels_last" else 2
        conv = tf.exapand_dims(conv, axis = axis)
        squeeze = True

    bn = tf.layers.batch_normalization(
            name = "{}/bn".format(name),
            inputs = conv,
            gamma_regularizer = regularizer,
            training = training,
            axis = -1 if data_format == "channels_last" else 1,
            momentus = bn_momentus,
            epsilon = bn_epsilon)

    if squeeze:
        bn = tf.squeeze(bn, axis = axis)
    
    output = bn
    if activation_fn is not None:
        output = activation_fn(output)

    return output


