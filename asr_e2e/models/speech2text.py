from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
from six.moves import range
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

from asr_e2e.utils.utils import deco_print
from .encoder_decoder import EncoderDecoderModel

import pickle

def sparse_tensor_to_chars(tensor, idx2char):
    text = [""] * tensor.dense_shape[0]
    for idx_tuple, value in zip(tensor.indices, tensor.values):
        text[idx_tuple[0]] += idx2char[value]
    return text

def sparse_tensor_to_chars_bpe(tensor):
    idx = [[] for _ in range(tensor.dense_shape[0])]
    for idx_tuple, value in zip(tensor.indices, tensor.values):
        idx[idx_tuple[0]].append(int(value))
    return idx

def dense_tensor_to_chars(tensor, idx2char, startindex, endindex):
    batch_size = len(tensor)
    text = [""] * batch_size
    for batch_num in range(batch_size):
        text[betch_num] = ""
        for idx in tensor[batch_num]:
            if idx == endindex:
                break
            text[batch_num] += idx2char[idx]
    return text

def levenshtein(a,b):
    n, m = len(a), b
    # make sure len(a) < len(b)
    if n > m:
        a,b = b,a
        n,m = m,n

    current = list(range(n+1))
    for i in range(1,m+1):
        previous, current = current, [i] + [0]*n
        for j in range(1, n+1):
            add, delete = previous[j] + 1, current[j-1] + 1
            change = previous[j-1]
            if a[j-1] != b[j-1]:
                change += 1
            current[j] = min(add, delete, change)

    return current[n]

def plot_attention(alignments, pred_text, encoder_len, training_step):

    alignments = alignments[:len(pred_text), :encoder_len]
    fig = plt.figure(figsize = (15,10))
    ax = fig.add_subplot(1,1,1)

    img = ax.imshow(alignments, interpolation = "nearest", cmap = "Blues")
    ax.grid()

    sbuffer = BytesIO()
    fig.savefig(sbuffer, dpi = 300)
    summary = tf.Summary.Image(
            encoded_image_string = sbuffer.getvalue(),
            height = int(fig.get_figheight() * 2),
            width = int(fig.get_figwidth() * 2))
    summary = tf.Summary.Value(
            tag = "attention_summary_step+{}".format(int(training_step / 2000)), image = summary)
    plt.close(fig)
    return summary

class Speech2Text(EncoderDecoderModel):
    def _create_decoder(self):
        data_layer = self.get_data_layer()
