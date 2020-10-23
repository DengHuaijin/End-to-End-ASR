from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import tensorflow as tf

from asr_e2e.utils.utils import get_base_config, check_base_model_logdir,\
                                check_logdir, create_model

def main():

    # Parse args and create config 
    args, base_config, base_model, config_module = get_base_config(sys.argv[1:])

    load_model = base_config.get('load_model', None)
    restore_best_checkpoint = base_config.get('restore_best_checkpoint', False)
    base_ckpt_dir = check_base_model_logdir(load_model, args, restore_best_checkpoint)
    base_config['load_model'] = base_ckpt_dir

    checkpoint = check_logdir(args, base_config, restore_best_checkpoint)

    # Create model and train/eval/infer
    with tf.Graph().as_default():
        model = create_model(args, base_config, config_module, base_model, checkpoint)
