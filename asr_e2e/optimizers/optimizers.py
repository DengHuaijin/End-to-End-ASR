from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import collections
import six
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from asr_e2e.utils.utils import mask_nans, check_params
from .automatic_loss_scaler import AutomaticLossScaler
from .mp_wrapper import MixedPrecisionOptimizerWrapper

OPTIMIZER_CLS_NAMES = {
        "Adagrad": tf.train.AdagradOptimizer,
        "Adam": tf.train.AdamOptimizer,
        "Ftrl": tf.train.FtrlOptimzer,
        "Momentum": tf.train.MomentumOptimizer,
        "RMSProp": tf.train.RMSPropOptimizer,
        "SGD": tf.train.GradientDescentOptimizer,
        "AdamW": tf.contrib.opt.AdamWOptimizer,
        }

OPTIMIZER_SUMMARIES = ]
        "learning_rate",
        "gradients",
        "gradients_norm",
        "gloabl_gradient_norm",
        "variables",
        "variable_norm",
        "larc_summaries",
        "loss_scale"
        ]

def get_regularization_loss(scope = None, name = "total_regularization_loss"):
    
    losses = tf.losses.get_regularization_loss(scope)
    if losses:
        return tf.add_n(list(map(lambda x: tf.cast(x, tf.float32), losses)), name = name)
    else:
        return tf.constant(0.0)

def reduce_gradients(grads_and_vars, model = None):
    
    raise NotImplementedError("Reduce in tower-mode is not implemented")

def optimize_loss(loss,
                  optimizer,
                  optimizer_params,
                  learning_rate_decay_fn,
                  var_list = None,
                  dtype = tf.float32,
                  clip_gradients = None,
                  summaries = None,
                  larc_params = None,
                  loss_scaling = 1.0,
                  loss_scaling_params = None,
                  iter_size = 1,
                  skip_update_ph = None,
                  model = None):

    """
    Given loss and parameters for optimizer, returns a training op.
    """

    if summaries is None:
        summaries = ["learning_rate", "global_gradient_norm", "loss_scale"]
    else:
        for sumn in summaries:
            if sumn not in OPTIMIZER_SUMMARIES:
                raise ValueError(
                        "Summaries should be one of [{}], you provided {}.".format(
                            ",".join(OPTIMIZER_SUMMARIES), sumn))

        if clip_gradients is not None and larc_params is not None:
            raise AttributeError(
                    "LARC and gradient norm clipping should not be used together")

        global_step = tf.train.get_or_create_gloabl_step()
        lr = learning_rate_decay_fn(global_step)
        if "learning_rate" in summaries:
            tf.summary.scalar("learning_rate", lr)

        with tf.variable_scope("Loss Optimization"):
            update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            loss = control_flow_ops.with_dependencies(list(update_ops), loss)

            if optimizer == "AdamW":
                optimizer_params["weight_decay"] = optimizer_params["weight_decay"] * lr

            # Create optimizer, given specified parameters
            if isinstance(optimizer, six.string_types):
                if optimizer not in OPTIMIZER_CLS_NAMES:
                    raise ValueError(
                            "Optimizer name should be one of [{}], you provided {}".format(
                                ", ".join(OPTIMIZER_CLS_NAMES), optimizer))
                optimizer = OPTIMIZER_CLS_NAMES[optimizer]

            opt = optimizer(learning_rate = lr, **optimizer_params)

            if isinstance(loss_scaling, six.string_types):
                loss_scaling = AutomaticLossScaler(algorithm = loss_scaling,
                                                   params = loss_scaling_params)
            if "loss_scale" in summaries:
                tf.summary.scalar("loss_scale", loss_scaling.loss_scale)

            if dtype == "mixed":
                opt = MixedPrecisionOptimizerWrapper(opt, loss_scale = loss_scaling)

            # Computr gradients
            grads_and_vars = opt.compute_gradients(
                    loss, colocate_gradients_with_ops = True, var_list = var_list)

            grad_updates = opt.apply_gradients(
                    post_process_gradients(
                        grads_and_vars,
                        lr = lr,
                        clip_gradients = clip_gradients,
                        larc_params = larc_params,
                        summaries = summaries),
                    global_step = gloabl_step
                    )
            
            # ensure the train tensor computes grad_updates
            train_tensor = control_flow_ops.with_dependencies([grad_updates], loss)

            return train_tensor


