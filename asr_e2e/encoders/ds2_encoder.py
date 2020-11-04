from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from six.moves import range

from asr_e2e.parts.cnns.conv_blocks import conv_bn_actv
from .encoder import Encoder

def rnn_cell(rnn_cell_dim, layer_type, dropout_keep_prob = 1.0):
    if layer_type == "layernorm_lstm":
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units = rnn_cell_dim, dropout_keep_prob = dropout_keep_prob)
    else:
        if layer_type == "lstm":
            cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_cell_dim)
        elif layer_type == "gru":
            cell = tf.nn.GRUCell(rnn_cell_dim)
        elif layer_type == "cudnn_gru":
            cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(rnn_cell_dim)
        elif layer_type == "cudnn_lstm":
            cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(rnn_cell_dim)
        else:
            raise ValueError("Error: not supported rnn type: {}".format(layer_type))

        cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_drop = dropout_keep_prob)

    return cell

def row_conv(name, input_layer, batch, channels, width, activation_fn, 
             regularizer, training, data_format, bn_momentum, bn_epsilon):

    if width < 2:
        return input_layer

    if data_format == "channels_last":
        x = tf.reshape(input_layer, [batch, -1, 1, channels])
    else:
        input_layer = tf.transpose(input_layer, [0,2,1]) # B C T
        x = tf.reshape(input_layer, [batch, channels, -1,1])

    cast_back = False

    if x.dtype.base_type == tf.float16:
        x = tf.cast(x, tf.float32)
        cast_back = True
    filters = tf.get_variable(
            name + "/w",
            shape = [width, 1, channels, 1],
            regularizer = regularizer,
            dtype = tf.float32)
    strides = [1,1,1,1]

    y = tf.nn.depthwise_conv2d(
            name = name + "/conv",
            input = x,
            filter = filters,
            strides = strides,
            padding = "SAME",
            data_format = "NHWC" if data_format == "channels_last" else "NCHW")
    
    bn = tf.layers.batch_normalization(
            name = "{}/bn".format(name),
            inputs = y,
            gamma_regularizer = regularizer,
            training = training,
            axis = -1 if data_format == "channels_last" else 1,
            momentum = bn_momentum,
            epsilon = bn_epsilon)

    output = activation_fn(bn)

    if data_format == "channels_first":
        output = tf.transpose(output, [0,2,3,1])
    output = tf.reshape(output, [batch, -1 ,channels])

    if cast_back:
        output = tf.cast(output, tf.float16)
    
    return output

class DeepSpeech2Encoder(Encoder):

    @staticmethod
    def get_required_params():
        return dict(Encoder.get_required_params(), **{
            "dropout_keep_prob": float,
            "conv_layers": list,
            "activation_fn": None,
            "num_rnn_layers": int,
            "row_conv": bool,
            "n_hidden": int,
            "use_cudnn_rnn": bool,
            "rnn_cell_dim": int,
            "rnn_type": ["layernorm_lstm", "lstm", "gru", "cudnn_gru", "cudnn_lstm"],
            "rnn_undirectional": bool})

    @staticmethod
    def get_optional_params():
        return dict(Encoder.get_optional_params(), **{
            "row_conv_width": int,
            "data_format": ["channels_first", "channels_last", "BCTF", "BCFT", "BTFC", "BFTC"],
            "bn_momentum": float,
            "bn_epsilon": float,})

    def __init__(self, params, model, name = "ds2_encoder", mode = "train"):
    """
    row_conv: whether to use a 'row' convolutional layer after RNNs
    n_hidden: last fully-connected layer
    """
        super(DeepSpeech2Encoder, self).__init__(params, model, name, mode)


    def _encode(self, input_dict):
        source_sequence, src_length = input_dict["source_tensors"]

        training= (self._model == "train")
        dropout_keep_prob = self.params["dropout_keep_prob"] if training else 0.0
        regularizer = self.params.get("regularizer", None)
        data_format = self.params.get("data_format", "channels_last")
        bn_momentum = self.params.get("bn_momentum", 0.99)
        bn_epsilon = self.params.get("bn_epsilon", 1e-3)

        input_layer = tf.expand_dims(source_sequence, axis = -1) # expand channels dim BTFC

        batch_size = input_layer.get_shape().as_list()[0]
        freq = input_layer.get_shape().as_list()[2]

        if data_format == "channels_last" or data_format == "BTFC":
            layout = "BTFC"
            dformat = "channels_last"
        elif data_format == "channels_first" or data_format == "BCTF":
            layout = "BCTF"
            dformat = "channels_first"
        elif data_format == "BFTC":
            layout = "BFTC"
            dformat = "channels_last"
        elif data_format == "BCFT":
            layout = "BCFT"
            dformat = "channels_first"
        else:
            print("WARNING: unsupported data format, will use channels_last (BTFC) instead")
            layout = "BTFC"
            dformat = "channels_last"

        if layout == "BCFT":
            top_layer = tf.transpose(input_layer, [0, 3, 1, 2])
        elif layout == "BTFC":
            top_layer = tf.transpose(input_layer, [0, 2, 1, 3])
        elif layout == "BCFT":
            top_layer = tf.transpose(input_layer, [0, 3, ,2, 1])
        else:
            top_layer = input_layer # BTFC

        conv_layers = self.params["conv_layers"]
        """
        CNN
        """
        for idx_conv in range(len(conv_layers)):
            ch_out = conv_layers[idx_conv]["num_channels"]
            kernel_size = conv_layers[idx_conv]["kernel_size"] # [T,F]
            strides = conv_layers[idx_conv]["stride"] # [T,F]
            padding = conv_layers[idx_conv]["padding"]

            if padding == "VALID":
                src_length = (src_length - kernel_size[0] + strides[0]) // strides[0]
                freq = (freq - kernel_size[1] + strides[1]) // strides[1]
            else:
                src_length = (src_length + strides[0] - 1) // strides[0]
                freq = (freq + strides[1] - 1) // strides[1]
                
            if layout == "BFTC" or layout == "BCFT":
                kernel_size = kernel_size[::-1]
                strides = strides[::-1] # default [T,F]

            top_layer = conv_bn_actv(
                    layer_type = "conv2d",
                    name = "conv{}".format(dix_conv + 1),
                    inputs = top_layer,
                    filters = ch_out,
                    kernel_size = kernel_size,
                    activation_fn = self.params["activation_fn"],
                    strides = strides,
                    padding = padding,
                    regularizer = regularizer,
                    training = training,
                    data_format = data_format,
                    bn_momentum = bn_momentum,
                    bn_epsilon = bn_epsilon)

        if layout == "BCTF": # BCTF -> BTFC
            top_layer = tf.transpose(top_layer, [0, 2, 3, 1])
        elif layout == "BFTC": # BFTC -> BTFC:
            top_layer = tf.transpose(top_layer, [0, 2, 1, 3])
        elif layout == "BCFT": # BCFT -> BCTF
            top_layer = tf.transpose(top_layer, [0, 3, 2, 1])

        # reshape to [B, T, F*C]
        f = top_layer.get_shape().as_list()[2]
        c = top_layer.get_shape().as_list()[3]
        fc = f * c
        top_layer = tf.reshape(top_layer, [batch_size, -1, fc])

        """
        RNN
        """
        num_rnn_layers = self.params["num_rnn_layers"]
        if num_rnn_layers > 0:
            rnn_cell_dim = self.params["rnn_cell_dim"]
            rnn_type = self.params["rnn_type"]
            if self.params["use_cudnn_rnn"]:
                # [B, T, C] -> [T, B, C]
                rnn_input = tf.transpose(top_layer, [1, 0 ,2])
                if self.params["rnn_undirectional"]:
                    direction = cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION
                else:
                    direction = cudnn_rnn_ops.CUDNN_RNN_BIDIRECRION

                if rnn_type == "cudnn_gru" or rnn_type == "gru":
                    rnn_block = tf.contrib.cudnn_rnn.CudnnGRU(
                            num_layers = num_rnn_layers,
                            num_units = rnn_cell_dim,
                            direction = direction,
                            dropout = 1.0 - dropout_keep_prob,
                            dtype = rnn_input.dtype,
                            name = "cudnn_gru")
                
                elif rnn_type == "cudnn_lstm" or rnn_type == "lstm":
                    rnn_block = tf.contrib.cudnn_rnn.CudnnLSTM(
                            num_layers = num_rnn_layers,
                            num_units = rnn_cell_dim,
                            direction = direction,
                            dropout = 1.0 - dropout_keep_prob,
                            dtype = rnn_input.dtype,
                            name = "cudnn_lstm")
                else:
                    raise ValueError(
                            "{} is not a valid rnn_type for cudnn_rnn layers".format(rnn_type))
            else:
                rnn_input = top_layer
                multirnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                        [rnn_cell(
                            rnn_cell_dim = rnn_cell_dim,
                            layer_type = rnn_type,
                            dropout_keep_prob = dropout_keep_prob)
                            for _ in range(num_rnn_layers)])
                
                if self.params["rnn_undirectional"]:
                    top_layer, state = tf.nn.dynamic_rnn(
                            cell = multirnn_cell_fw,
                            inputs = rnn_input,
                            sequence_length = src_length,
                            dtype = rnn_input.dtype,
                            time_major = False)
                else:
                    multirnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                            [rnn_cell(
                                rnn_cell_dim = rnn_cell_dim,
                                layer_type = rnn_type,
                                dropout_keep_prob = dropout_keep_prob)
                                for _ in range(num_rnn_layers)])

                    top_layer, state = tf.nn.bidirectional_dynamic_rnn(
                            cell_fw = multirnn_cell_fw,
                            cell_bw = multirnn_cell_bw,
                            inputs = rnn_input,
                            sequence_length = src_length,
                            dtype = rnn_input.dtype,
                            time_major = False)
                    
                    # [B, T, n_cell_dim] -> [B, T, 2*n_cell_dim]
                    top_layer = tf.concat(top_layer, 2)
        
        if self.params["row_conv"]:
            channels = top_layer.get_shape().as_list()[-1]
            top_layer = row_conv(
                    name = "row_conv",
                    input_layer = top_layer,
                    batch = batch_size,
                    channels = channels,
                    activation_fn = self.params["activation_fn"],
                    width = self.params["row_conv_width"],
                    regularizer = regularizer,
                    training = training,
                    data_format = data_format,
                    bn_momentum = bn_momentum,
                    bn_epsilon = bn_epsilon)
        
        # [B, T, C] -> [ B*T, C]
        c = top_layer.get_shape().as_list()[-1]
        top_layer = tf.reshape(top_layer, [-1, c])

        top_layer = tf.layers.dense(
                inputs = top_layer,
                units = self.params["n_hidden"],
                kernel_regularizer = regularizer,
                activation = self.params["activation_fn"],
                name = "fully_connected")
        outputs = tf.nn.dropout(x = top_layer, keep_prob = dropout_keep_prob)
        
        # [B*T, A] -> [B, T, A]
        outputs = tf.reshape(outputs, [batch_size, -1, self.params["n_hidden"]])

        return {"outputs": outputs, "src_length": src_length}