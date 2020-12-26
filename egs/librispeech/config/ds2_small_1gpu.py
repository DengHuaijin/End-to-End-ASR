import tensorflow as tf
from asr_e2e.models import Speech2Text
from asr_e2e.encoders import DeepSpeech2Encoder
from asr_e2e.decoders import FullyConnectedCTCDecoder
from asr_e2e.data import Speech2TextDataLayer
from asr_e2e.losses import CTCLoss
from asr_e2e.optimizers.lr_policies import exp_decay

base_model = Speech2Text

base_params = {

    # "load_model": "egs/librispeech/ds2_log",

    "random_seed": 0,
    "num_epochs": 12,
    
    "num_gpus": 1,
    "batch_size_per_gpu": 32,
    
    "save_summaries_steps": 100,
    "print_loss_steps": 10,
    "print_samples_steps": 5000,
    "eval_steps": 5000,
    "save_checkpoint_steps": 1000,
    "logdir": "egs/librispeech/ds2_log",

    "optimizer": "Adam",
    "optimizer_params": {},
    "lr_policy": exp_decay,
    "lr_policy_params": {
        "learning_rate": 0.0001,
        "begin_decay_at": 0,
        "decay_steps": 5000,
        "decay_rate": 0.9,
        "use_staircase_decay": True,
        "min_lr": 0.0},
    "dtype": tf.float32,
    "regularizer": tf.contrib.layers.l2_regularizer,
    "regularizer_params": {"scale": 0.0005},

    "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                  'variable_norm', 'gradients_norm', 'gloabl_gradient_norm'],

    "initializer": tf.contrib.layers.xavier_initializer,
    
    "encoder": DeepSpeech2Encoder,
    "encoder_params":{
        "conv_layers":[
            {
                "kernel_size": [11,41],
                "stride": [2,2],
                "num_channels": 32,
                "padding": "SAME"
            },
            
            {
                "kernel_size": [11,21],
                "stride": [1,2],
                "num_channels": 32,
                "padding": "SAME"
            }],
        "data_format": "channels_first", # batch first channel last
        "n_hidden": 1024,

        "rnn_cell_dim": 512,
        "rnn_type": "cudnn_gru",
        "num_rnn_layers": 2,
        "rnn_unidirectional": False,
        "row_conv": False,
        "use_cudnn_rnn": True,

        "dropout_keep_prob": 0.5,
        "activation_fn": tf.nn.relu,
        },

    "decoder": FullyConnectedCTCDecoder,
    "decoder_params": {
        "use_language_model" : False,

        "beam_width": 512,
        "alpha": 2.0,
        "beta": 1.0,

        "decoder_library_path": "ctc_decoder_with_lm/libctc_decoder_with_kenlm.so",
        "trie_path": "language_model/trie.binary",
        "lm_path": "language_model/4-gram.binary",
        "alphabet_config_path": "egs/librispeech/config/vocab.txt",
            },
    "loss": CTCLoss,
    "loss_params": {},
}

train_params = {
        "data_layer": Speech2TextDataLayer,
        "data_layer_params": {
            "num_audio_features": 96,
            "input_type": "spectrogram",
            "augmentation": {
                "speech_perturbation_ratio": 0.05,
                "noise_level_min": -90,
                "noise_level_max": -60,
            },

            "vocab_file": "egs/librispeech/vocab.txt",
            "dataset_files": [
                "egs/librispeech/data/librispeech/librivox-train-clean-100.csv",
                "egs/librispeech/data/librispeech/librivox-train-clean-360.csv"
                ],
            "shuffle": True,
            }
        }

eval_params = {
        "data_layer": Speech2TextDataLayer,
        "data_layer_params": {
            "num_audio_features": 96,
            "input_type": "spectrogram",
            "vocab_file": "egs/librispeech/vocab.txt",
            "dataset_files": [
                "egs/librispeech/data/librispeech/librivox-dev-clean.csv",
                ],
            "shuffle": False
            }
        }   

