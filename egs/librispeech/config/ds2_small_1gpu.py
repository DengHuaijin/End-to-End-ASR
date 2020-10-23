import tensorflow as tf
from asr_e2e.models import Speech2Text
from asr_e2e.encoders import DeepSpeech2Encoder
from asr_e2e.decoders import FullyConnectedCTCDecoder
from asr_e2e.data import Seepch2TextDataLayer
from asr_e2e.losses import CTCLoss
from asr_e2e.optimizers.lr_polices import poly_decay

base_model = Speech2Text

base_params = {
        "random_seed": 0,
        "mxa_steps": 1000,
        
        "num_gpus": 2,
        "batch_size_per_gpu": 8,
        
        "save_summaries_steps": 100,
        "print_loss_steps": 100,
        "print_samples_steps": 100,
        "eval_steps": 500,
        "save_checkpoint_steps": 500,
        "logdir": "ds2_log",

        "optimizer": "Momentum",
        "optimizer_params": {
            "momentum": 0.9},
        "lr_policy": poly_decay,
        "lr_policy_params": {
            "learning_rate": 0.001,
            "power": 2},
        "larc_params": {
            "larc_eta": 0.001},
        "dtype": tf.float32

        "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                      'varibale_norm', 'gradients_norm', 'gloabl_gradient_norm'],

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
                    "num_channels": 96,
                    "padding": "SAME"
                }],
            "data_format": "BFTC", # batch first channel last
            "n_hidden": 256,

            "rnn_cell_dim": 256,
            "rnn_type": "gru",
            "num_rnn_layers": 1,
            "run_undirectional": False,
            "row_conv": False,
            "row_conv_width": 8,
            "use_cudnn_rnn": True,

            "dropout_keep_prob": 1.0,

            "initializer": tf.contrib.layers.xavier_initilaizer,
            "initializer_params": {
                'uniform': False
                },
            "activation_fn": lambda x: tf.minimum(tf.nn.relu(x), 20.0),

            },

        "decoder": FullyConnectedCTCDecoder,
        "decoder_params": {
                "initializer": tf.contrib.layers.xavier_initilaizer,
                "use_language_model" : False,

                "beam_width": 64,
                "alpha": 1.0,
                "beta": 1.5,

                "decoder_library_path": "ctc_decoder_with_lm/libctc_decoder_with_kenlm.so",
                "lm_path": "language_model/trie.binary",
                "alphabet_config_path": "vocab.txt",
                },
        "loss": CTCLoss,
        "loss_params": {},
}

eval_params = {
        "data_layer": Speech2TextDataLayer,
        "data_layer_params": {
            "num_audio_features": 160,
            "input_type": "spectrogram",
            "vocab_file": "vocab.txt",
            "dataset_files": [
                "data.csv"
                ],
            "shuffle": False
            }
}


infer_params = {
        "data_layer": Speech2TextDataLayer,
        "data_layer_params": {
            "num_audio_features": 160,
            "input_type": "spectrogram",
            "vocab_file": "vocab.txt",
            "dataset_files": [
                "data.csv"
                ],
            "shuffle": False
            }
}
