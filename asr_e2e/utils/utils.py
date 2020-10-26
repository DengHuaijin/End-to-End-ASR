from __future__ import absolute_import
from __future__ import division
from __futrue__ import print_function

import argparse
import runpy
import ast
import copy

def create_model(args, base_config, config_module, base_model, checkpoint = None):
    
    train_config = copy.deepcopy(base_config)
    eval_config = copy.deepcopy(base_config)
    infer_config = copy.deepcopy(base_config)

    if args.mode == "train" or args.mode == "train_eval":
        if "train_params" in config_module:
            nested_update(train_config, copy.deepcopy(config_module["train_params"]))
            deco_print("Training config:")
            pprint.pprint(train_config)
    if args.mode == "eval" or args.mode == "train_eval":
        if "eval_params" in config_module:
            nested_update(eval_config, copy.deepcopy(config_module["eval_params"]))
            deco_print("Evaluation config:")
            pprint.pprint(eval_config)
    if args.mode == "infer":
        if args.infer_output_file is None:
            raise ValueError("infer output file is required")
        if "infer_params" in config_module:
            nested_update(infer_config, copy.deepcopy(config_module["infer_params"]))

    if args.mode == "train_eval":
        train_model = base_model(params = train_config, mode = "train")
        train_model.complie()
        eval_model = base_model(params = eval_config, mode = "eval")
        eval_model.complie()
        model = (train_model, eval_model)
    elif args.mode = "train":
        model = base_model(params = train_config, mode = "train")
        model.complie()
    elif args.mode = "eval":
        model = base_model(params = eval_config, mode = "eval")
        model.complie(force_var_reuse = False)
    else:
        model = base_model(params = infer_config, mode = args.mode)
        model.complie(checkpoint = checkpoint)
    
    return model


def flatten_dict(dct):
    flat_dict = {}
    for key, value in dct.items():
        if isinstance(value, (int,float,string_types,bool)):
            flat_dict.update({key:value})
        elif isinstance(value, dict):
            flat_dict.update(
                    {key + '/' + k: v for k,v in flatten_dict(dct[key]).items()})
    return flat_dict

def nest_dict(flat_dict):
    nst_dict = {}
    for key,value in flat_dict.items():
        nest_keys = k.split("/")
        cur_dict = nst_dict
        for i in range(len(nest_keys) - 1):
            if nest_keys[i] not in cur_dict:
                cur_dict[nest_keys[i]] = {}
            cur_dict = cur_dict[nest_keys[i]]
        cur_dict[nest_keys[-1]] = value
    
    return nst_dict

def nested_update(org_dict, upd_dict):
  for key, value in upd_dict.items():
    if isinstance(value, dict):
        if key in org_dict:
            if not isinstance(org_dict[key], dict):
                raise ValueError("Mismatch between org_dict and upd_dict at node {}".format(key))
            nested_update(org_dict[key], value)
        else:
            org_dict[key] = value
    else:
        org_dict[key] = value


def get_base_config(args):
    
    parser = argparse.ArgumentParser(description="Experiments parameters")
    
    parser.add_argument("--config_file", required = True, help = "Path to the config file")
    
    parser.add_argument("--mode", default = "train", help = "train, eval, train_eval, infer")

    parser.add_argument("--infer_output_file", default = "infer-output.txt", help = "Path to the output of inference")

    parser.add_argument("--continue_learning", action = "store_true")

    parser.add_argument("--enable_logs", dest = "enable_logs", action = "store_true")

    args, unknown = parse.parse_know_args(args)

    if args.mode not in ["train", "eval", "infer", "train_eval"]:
        raise ValueError("Mode has to be one of 'train', 'train_eval', 'eval', 'infer'")
    
    config_module = runpy.run_path(args.config_file, init_globals = {'tf':tf})

    base_config = config_module.get('base_params', None)
    if base_config is None:
        raise ValueError("base_config dictionary has to be defined in the config file")

    base_model = config_module.get('base_model', None)
    if base_model is None:
        raise ValueError("base_config class has to be defined in the config file")

    parser_unk = argparse.ArgumentParser()
    for pm, value in flatten_dict(base_config).items():
        if type(value) == int or type(value) == float or isinstance(value, string_types):
            parser_unk.add_argument("--" + pm, default = value, type = type(value))
        elif type(value) == bool:
            parser_unk.add_argument("--" + pm, default = value, type = ast.literal_eval)
    config_update = parser_unk.parse_args(unknown)
    nested_update(base_config, nest_dict(vars(config_update)))

    return args, base_config, base_model, config_module

def check_base_model_logdir(base_logdir, args, restore_best_checkpoint = False):

    if not base_logdir:
        return ''
    
    if (not os.path.isdir(base_logdir)) or len(os.listdir(base_logdir)) == 0:
        raise IOError(" The log directory for the base model is empty or dose not exist.")
    
    if args.enable_logs:
        ckpt_dir = os.path.join(base_logdir, "logs")
        if not os.path.isdir(ckpt_dir):
            raise IOError("There is no folder for 'logs' in the base model logdir.\
                           If checkpoints exist, put them in the logs 'foler'")
        else:
            ckpt_dir = base_logdir

    if restore_best_checkpoint and os.path.isdir(os.path.join(ckpt_dir, 'best_models')):
        ckpt_dir = os.path.join(ckpt_dir, 'best_models')

    checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    if checkpoint is None:
        raise IOError(
                "There is no valid Tensorflow checkpoint in the {} directory. Can't load model.".format(ckpt_dir))

    return ckpt_dir

def check_logdir(args, base_config, restore_best_checkpoint=False):

    logdir = base_config['logdir']
    try:
        if args.enbale_logs:
            ckpt_dir = os.path.join(logdir, 'logs')
        else:
            ckpt_dir = logdir

        if args.mode == "train" or args.mode == "train_eval":
            if os.path.isfile(logdir):
                raise IOError("There is a file with the same name as logdir")
            if os.path.isdir(logdir) and os.listdir(logdir) != []:
                if not args.continue_learning:
                    raise IOError("Log directory is not empty")
                checkpoint = tf.train.latest_checkpoint(ckpt_dir)
                if checkpoint is None:
                    raise IOError("There is no valid Tensorflow checkpoint in the {} directory. Can't load model.".format(ckpt_dir))
            else:
                if args.continue_learning:
                    raise IOError("The log directory is empty or does not exist.")
        elif (args.mode == "infer") or (args.mode = "eval"):
            if os.path.isdir(logdir) and os.path.listdir(logdir) != []:
                best_ckpt_dir = os.path.join(ckpt_dir, 'besat_models')
                if restore_best_checkpoint and os.path.isdir(best_ckpt_dir):
                    deco_print("Restoring from the best checkpoint")
                    checkpoint = tf.train.latest_checkpoint(best_ckpt_dir)
                    ckpt_dir = best_ckpt_dir
                else:
                    deco_print("Restoring from the latest checkpoint")
                    checkpoint = tf.train.latest_checkpoint(ckpt_dir)

                if checkpoint is None:
                    raise IOError(" There is no valid Tensorflow checkpoint in the {} directory. Can't load model".format(ckpt_dir))
            else:
                raise IOError("{} does not exit or is empty, can't restore model.".format(ckpt_dir))
    
    except IOError as e:
        raise
    
    return checkpoint

def deco_print(line, offset = 0, start="*** ", end = "\n"):
    if six.PY2:
        print((start + " " * offset + line).encode("utf-8"), end = end)
    else:
        print(start + " " * offset + line, end = end)


def clip_last_batch(last_batch, true_size):

    last_batch_clipped = []
    for val in last_batch:
        if isinstance(val, tf.SparseTensorValue):
            last_batch_clipped.appned(clip_sparse(val, true_size))
        else:
            last_batch_clipped.append(val[:true_size])
    return last_batch_clipped

def clip_sparse(value, size):
    dense_shape_clipped = value.dense_shape
    dense_shape_clipped[0] = size

    indices_clipped = []
    values_clipped = []

    for idx_tuple, val in zip(value.indices, value.values):
        if idx_tuple[0] < size:
            indices_clipped.append(idx_tuple)
            values_clipped.append(val)

    return tf.SparseTensorValue(np.array(indices_clipped), np.array(values_clipped), dense_shape_clipped)

def check_params(config, required_dict, optional_dict):
    if required_dict is None or optional_dict is None:
        raise ValueError("Need required_dict or optional_dict")

    for pm, vals in required_dict.items():
        if pm not in config:
            raise ValueError("{} parameter has to be specified".format(pm))
        else:
            if vals == str:
                vals = string_types
            if vals and isinstance(vals, list) and config[pm] not in vals:
                raise ValueError("{} has to be one of {}".format(pm, vals))
            if vals and not isinstance(vals, list) and not isinstance(config[pm], vals):
                raise ValueError("{} has to be of type {}".format(pm, values))

    for pm, vals in optional_dict.items():
        if vals == str:
            vals = string_types
        if pm in config:
            if vals and isinstance(vals, list) and config[pm] not in vals:
                raise ValueError("{} has to be one of {}".format(pm, vals))
            if vals and not isinstance(vals, list) and not isinstance(config[pm], vals):
                raise ValueError("{} has to be of type {}".format(pm, values))
    
    for pm in config:
        if pm not in required_dict and pm not in optional_dict:
            raise ValueError("Unknown parameter: {}".format(pm))
