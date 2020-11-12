from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import io

from six.moves import range

def pad_vocab_to_eight(vocab):
    """
    Pads vocabulary so that it is divisible by 8

    vocab: dict in the form token->id
    """

    v_len = len(vocab)
    if v_len % 8 == 0:
        return vocab

    for id_add in range(0, 8 - v_len % 8):
        vocab["<$" + str(id_add) + "$>"] = v_len + id_add

    return vocab

def load_pre_existing_vocabulary(path, min_idx = 0, read_chars = False):
    """
    Returns
        dict: vocabulary dictionary mapping tokens to int ids
    """

    idx = min_idx
    vocab_dict = {}
    with io.open(path, newline = "", encoding = "utf-8") as f:
        for line in f:
            if not line or line == "\n":
                continue
            if read_chars:
                token = line[0]
            else:
                token = line.rstrip().split("\t")[0]
            vocab_dict[token] = idx
            idx += 1
    
    return vocab_dict
