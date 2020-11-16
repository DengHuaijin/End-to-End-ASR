from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import math
import tensorflow as tf

def exp_decay(global_step, learning_rate,
              decay_steps,decay_rate, use_staircase_decay, 
              begin_decay_at = 0, min_lr = 0):
    """
    Equivalent to tf.train.exponential_decay with some
    additional functionality.

    """

    """
    tf.conf(pred, true_fn, false_fn)
    if pred: true_fn() else false_fn()
    """
    new_lr = tf.cond(
            global_step < begin_decay_at,
            lambda: learning_rate,
            lambda: tf.train.exponential_decay(
                learning_rate = learning_rate,
                global_step = global_step - begin_decay_at,
                decay_steps = decay_steps,
                decay_rate = decay_rate,
                staircase = use_staircase_decay),
            name = "learning_rate",)
    final_lr = tf.maximum(min_lr, new_lr)
    return final_lr
