from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tensorflow as tf

from asr_e2e.utils.utils import deco_print, get_base_config, check_base_model_logdir, create_logdir, check_logdir, create_model

from asr_e2e.utils.funcs import train, evaluate

def main():

    """
    Parse args and create config 
    e.g. python3 run.py --mode=train --config_file=config/ds2_small_1gpu.py
    """
    import sys
    args, base_config, base_model, config_module = get_base_config(sys.argv[1:])

    load_model = base_config.get('load_model', None)
    restore_best_checkpoint = base_config.get('restore_best_checkpoint', False)
    base_ckpt_dir = check_base_model_logdir(load_model, args, restore_best_checkpoint)
    base_config['load_model'] = base_ckpt_dir

    checkpoint = check_logdir(args, base_config, restore_best_checkpoint)

    if args.enable_logs:
        old_stdout, old_stderr, stdout_log, stderr_log = create_logdir(args, base_config)
        base_config["logdir"] = os.path.join(base_config["logdir"], 'logs')

    if args.mode == "train":
        if checkpoint is None:
            if base_ckpt_dir:
                deco_print("Starting training from the base model")
            else:
                deco_print("Starting training from scratch")
        else:
            deco_print("Resroring checkpoint from {}".format(checkpoint))

    elif args.mode == "eval":
        deco_print("Loading model from {}".format(checkpoint))

    # Create model and train/eval
    with tf.Graph().as_default():
        model = create_model(args, base_config, config_module, base_model, checkpoint)
        
        if args.mode == "train":
            train(model, eval_model = None, debug_port = args.debug_port, custom_hooks = None)
        elif args.mode == "eval":
            evluate(model, checkpoint)

    if args.enable_logs:
        sys.stdout = old_stdout
        sys,stderr = old_stderr
        stdout_log.close()
        stderr_log.close()

if __name__ == "__main__":
    main()
