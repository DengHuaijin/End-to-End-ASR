from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import time

import numpy as np
import tensorflow as tf

from six.moves import range

from .utils import deco_print, get_results_for_epoch
"""
from .hooks import PrintSampleHook, RunEvaluationHook, PrintLossAndTimeHook, BroadcastGlobalVariablesHook
from .helpers import TransferMonitoredTrainingSession, TransferScaffold, \
                     get_assign_ops_and_restore_dict, run_assign_and_saver
from asr_e2e.data import WKTDataLayer
"""

def train(train_model, eval_model = None, debug_report = None):
    
    sess_config = tf.ConfigProto(allow_soft_placement = True)
    sess_config.gpu_options.allow_growth = True

    checkpoint_dir = train_model.params["logdir"]
    load_model_dir = train_model.params["load_model"]

    if train_model.params["save_checkpoint_steps"] is not None:
        saver = tf.train.Saver(save_relative_paths = True, max_to_keep = train_model.params["num_checkpoints"])

    total_time = 0.0

    init_data_layer = tf.group([train_model.get_data_layer(i).iterator.initializer for i in range(train_model.num_gpus)])

    restoring = load_model_dir and not tf.train.latest_checkpoint(checkpoint_dir)
    if restoring:
        vars_in_checkpoint = {}
        for var_name, var_shape in tf.train.list_variables(load_model_dir):
            vars_in_checkpoint[var_name] = var_shape

        print("VARS_IN_CHECKPOINT:")
        print(vars_in_checkpoint)

        vars_to_load = []
        for var in tf.global_variables():
            var_name = var.name.split(":")[0]
            if var_name in vars_in_checkpoint:
                if var.shape == vars_in_checkpoint[var_name] and "global_step" not in var_name:
                    vars_to_load.append(var)

        print("VARS_TO_LOAD:")
        for var in vars_to_load:
            print(var)

        load_model_fn = tf.contrib.framework.assign_from_checkpoint_fn(
                tf.train.latest_checkpoint(load_model_dir), vars_to_load)

        scaffold = tf.train.Scaffold(
                local_init_op = tf.group(tf.local_variables_initializer(), init_data_layer),
                init_fn = lambda scaffold_self, sess: load_model_fn(sess))

    else:
        scaffold = tf.train.Scaffold(
                local_init_op = tf.group(tf.local_variables_initializer(), init_data_layer))
    fetches = [train_model.train_op]

    try:
        total_objects = 0.0
        for worker_id in range(train_model.num_gpus):
            fetches.append(train_model.get_num_objects_per_step(worker_id))
    except NotImplementedError:
        deco_print("WARNING: Can't compute number of objects per step, since train model does not define get_num_objects_per_step method")

    
    # Starting training
    sess = tf.train.MonitoredTrainingSession(
            scaffold = scaffold,
            checkpoint_dir = checkpoint_dir,
            save_summaries_steps = train_model.params["save_summaries_steps"],
            config = sess_config,
            save_checkpoint_secs = None,
            log_step_count_steps = train_model.params["save_summaries_steps"],
            stop_grace_period_secs = 300,
            hooks = None)
    step = 0
    num_bench_updates = 0

    while True:
        if sess.should_stop():
            break
        try:
            feed_dict = []
            iter_size = train_model.params.get("iter_size", 1)
            if iter_size > 1:
                feed_dict[train_model.skip_update_ph] = step % iter_size != 0
            if step % iter_size == 0:
                fetches_vals = sess.run(fetches, feed_dict)
            else:
                # necessary to skip "no-update" steps when iter_size > 1
                def run_with_no_hooks(step_context):
                    return step_context.session.run(fetches, feed_dict)
                fetches_vals = sess.run_step_fn(run_with_no_hooks)
        except tf.errors.OutOfRangeError:
            break

        step += 1
    sess.close()

    deco_print("Finished training")

def restore_and_get_results(model, checkpoint, mode):
    
    sess_config = tf.ConfigProto(allow_soft_placement = True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config = sess_config) as sess:
        results_per_batch = get_results_for_epoch(model, sess, mode = mode, compute_loss = False, verbose = True)

    return results_per_batch

def evaluate(model, checkpoint):
    results_per_batch = restore_and_get_results(model, checkpoint, mode = "eval")
    eval_dict = model.finalize_evaluation(results_per_batch)
    deco_print("Fnished evaluation")

    return eval_dict
