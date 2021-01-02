import librosa

import numpy as np
import argparse
import scipy.io.wavfile as wave
import tensorflow as tf

from asr_e2e.utils.utils import deco_print, get_base_config, check_logdir,\
                                create_logdir, create_model, get_interactive_infer_results

parser = argparse.ArgumentParser()
parser.add_argument("--input_audio", required = True)
args = parser.parse_args()

input_audio = args.input_audio

asr_args = ["--config_file=egs/librispeech/config/ds2_small_1gpu.py",
        "--mode=interactive_infer",
        "--logdir=egs/librispeech/ds2_log",
        "--batch_size_per_gpu=1"]

with tf.variable_scope("ds2_model"):
    args, base_config, base_model, config_module = get_base_config(asr_args)
    checkpoint = check_logdir(args, base_config)
    model = create_model(args, base_config, config_module, base_model, None)

sess_config = tf.ConfigProto(allow_soft_placement = True)
sess_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config = sess_config)

vars_model = {}

for v in tf.get_collection(tf.GraphKeys.VARIABLES):
    print(v)
    vars_model["/".join(v.op.name.split("/")[1:])] = v

saver = tf.train.Saver(vars_model)
saver.restore(sess, checkpoint)

# wav = librosa.core.resample(wavfile, sr, 16000)
model_input = input_audio
result = get_interactive_infer_results(model, sess, model_input = [model_input])
words = result[0][0]

print("######## Recognized words: ########")
print(words)
print("###################################")
