from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import tensorflow as tf

from asr_e2e.utils.utils import mask_nans, deco_print
from .loss import Loss

def dense_to_sparse(dense_tensor, sequence_length):
    """
    SparseTensor就是用来描述稀疏矩阵的一种数据Tensor
    对一个维度是dense_shape的稀疏矩阵，用indices标记出
    非零元素的位置，用value来存储这些数值：
    SparseTensor(indices = [[0,0], [1,2]], values = [1,2], dense_shape = [3,4])
    [[1,0,0,0],
     [0,0,2,0],
     [0,0,0,0]]
    indices: A 2-D int64 tensor of shape[N, ndims]
        N: the number of values
    
    values: A 1-D tensor of any type and shape [N], which supplies the values for each element in indices.
    e.g. indices = [[1,3], [2,4]] values = [18, 3.6] -> the element [1,3] of the sparse tensor
    has a value of 18

    dense_shape: A 1-D int64 tensor of shape [ndims].
    e.g. dense_shape = [3,6] specifies a 2-dimensional 3x6 tensor
         dense_shape = [2,3,4] specifies a 3-dimensional 2x3x4 tensor
    """
    indices = tf.where(tf.sequence_mask(sequence_length))
    values = tf.gather_nd(dense_tensor, indices)
    shape = tf.shape(dense_tensor, out_type = tf.int64)

    return tf.SparseTensor(indices, values, shape)

class CTCLoss(Loss):

    @staticmethod
    def get_optional_params():
        return dict(
                Loss.get_optional_params(), **{"mask_nan": bool})

    def __init__(self, params, model, name = "ctc_loss"):

        super(CTCLoss, self).__init__(params, model, name)
        self._mask_nan = self.params.get("mask_nan", True)
        self.params["dtype"] = tf.float32

    def _compute_loss(self, input_dict):
        """
        Inputs
            input_dict = {"decoder_output":{
                                "logits": tensor, shape [batch_size, time_length, tgt_vocab_size]
                                "sec_length": tensor, shape [batch_size]},
                          "target_tensor": [
                            tgt_sequence shape [batch_size, time_length, num_features],
                            tgt_length shape [batch_size]]
                         }
        Return
            Average CTC Loss
        """

        logits = input_dict["decoder_output"]["logits"]
        tgt_sequence, tgt_lenght = input_dict["target_tensors"]
        # ctc loss needs an access to src_length, since they might be changed in the encoder
        src_length = input_dict["decoder_output"]["src_length"]

        total_loss = tf.nn.ctc_loss(
                labels = dense_to_sparse(tgt_sequence, tgt_length),
                inputs = logits,
                sequence_length = src_length,
                ignore_longer_outputs_than_inputs = True)

        if self._mask_nan:
            total_loss = mask_nans(total_loss)
        
        avg_loss = tf.reduce_mean(total_loss)
        return avg_loss

