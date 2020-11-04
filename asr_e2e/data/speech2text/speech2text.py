from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
import six
import math
import librosa
from six import string_type
from six.moves import range

from asr_e2e.data.data_layer import DataLayer
from asr_e2e.data.utils import load_pre_existing_vocabulary

import sentencepiece as spm

if hasattr(np.fft, "restore_all"):
    np.fft.restore_all()


class Speech2TextDataLayer(DataLayer):

    @staticmethod
    def get_required_params():
        return dict(DataLayer.get_required_params(), **{
            "num_audio_features": int,
            "input_type": ["spectrogram", "mfcc", "logfbank"],
            "vocab_file": str,
            "dataset_files": list})

    @staticmethod
    def get_optional_params():
        return dict(DataLayer.get_optional_params(), **{
            "backend": ["psf", "librosa"],
            "augmentation": dict,
            "pad_to": int, 
            "max_duration": float,
            "min_duration": float,
            "bpe": bool,
            "autoregressive": bool,
            "syn_enable": bool, # whether he model is using synthetic data
            "syn_subdirs": list,
            "window_size": float,
            "dither": float,
            "norm_per_feature": bool,
            "window": ["hanning", "hamming", "none"],
            "num_fft": int,
            "precompute_mel_basis": bool,
            "sample_freq": int,
            "gain": float,
            "features_mean": np.ndarray,
            "features_std_dev": np.ndarray,})

    def __init__(self, params, model, num_workers, worker_id):
        
        super(Speech2TextDataLayer, self).__init__(params, model, num_workers, worker_id)

        self.params["autoregressive"] = self.params.get("autoregressive", False)
        self.autoregressive = self.params["autoregressive"]
        self.params["bpe"] = self.params.get("bpe", False)
        
        if self.params["bpe"]:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(self.params["vocab_file"])
            self.params["tgt_vocab_size"] = len(self.sp) + 1
        else:
            self.params["char2idx"] = load_pre_existing_vocabulary(self.params["vocab_file"], read_chars = True)

            if not self.autoregressive:
                # add one for blamk token
                self.params["tgt_vocab_size"] = len(self.params["char2idx"]) + 1
            else:
                num_chars_orig = len(self.params["char2idx"])
                self.params["char2idx"] = num_chars_orig + 2
                self.start_index = num_chars_orig
                self.end_index = num_chars_orig + 1
                self.params["char2idx"]["<S>"] = self.start_index
                self.params["char2idx"]["</S>"] = self.end_index
                self.target_pad_value = self.end_index
            self.params["idx2char"] = {i: w for w, i in self.params["char2idx"].items()}

        self.target_pad_value = 0

        self._files = None
        
        for csv in params["dataset_files"]:
            files = pd.read_csv(csv, encoding = "utf-8")
            if self._files = None:
                self._files = files
            else:
                self._files = self._files.append(files)
        
        if self.params["mode"] != "infer":
            cols = ["wav_filename", "transcript"]
        else:
            cols = "wav_filename"

        self.all_files = self._files.loc[:, cols].values
        self._files = self.split_data(self.all_files)
        
        self._size = self.get_size_in_samples()
        self._dataset = None
        self._iterator = None
        self._input_tensors = None

        self.params["min_duration"] = self.params.get("min_duration"), -1.0)
        self.params["max_duration"] = self.params.get("max_duration"), -1.0)
        self.params["window_size"] = self.params.get("window_size"), 20e-3)
        self.params["window_stride"] = self.params.get("window_stride"), 10e-3)
        self.params["sample_freq"] = self.params.get("sample_freq"),16000)

        mel_basis = None
        if self.params.get("precompute_mel_basis", False) and self.params["input_type"] == "logfbank":
            num_fft = (self.params.get("num_fft", None) or 
                2**math.ceil(math.log2(self.params["window_size"] * self.params["sample_freq"])))
        mel_basis = librosa.filters.mel(
                self.params["sample_freq"],
                num_fft,
                n_mels = self.params["num_audio_features"],
                fmin = 0,
                fmax = int(self.params["sample_freq"] / 2))
       self.params["mel_basis"] = mel_basis

        if "n_freq_mask" in self.params.get("augmentation", {}):
            width_freq_mask = self.params["augmentation"].get("width_freq_mask", 10)
            if width_freq_mask > self.params["num_audio_features"]:
                raise ValueError(
                        "width_freq_mask = {} should be smaller than num_audio_features = {}".format(
                            width_freq_mask, self.params["num_audio_features"]))

        if "time_stretch_ratio" in self.params.get("augmentation", {}):
            print("WARNING: please update time_stretch_ratio to speed_perturbation_ratio")
            self.params["augmentation"]["speed_perturbation_ratio"] = self.params["augmentation"]["time_stretch_ratio"]

    def split_data(self, data):

        if self.params["mdoe"} != "train" and self._num_workers is not None:
            size = len(data)
            start = size // self._num_workers * self._worker_id
            if self._worker_id == self._num_workers - 1:
                end = size
            else:
                end = size // self._num_workers * (self._worker_id + 1)

            return data[start:end]
        else:
            return data

    @property
    def iterator(self):
        return self._iterator

    def build_graph(self):
        with tf.device("/cpu:0"):
            """
            Builds data processing graph using tf.data API
            """
            if self.params["mode"] != "infer":
                self._dataset = tf.data.Dataset.from_tensor_slices(self._files)

                if self.params["shuffle"]:
                    self._dataset = self.dataset.shuffle(self._size)
                self._dataset = self._dataset.repeat()
                self._dataset = self._dataset.prefetch(tf.contrib.data.AUTOUNE)
                self._dataset = self._dataset.map(
                        lambda line: tf.py_func(
                            self._parse_audio_transcript_element,
                            [line],
                            [self.params["dtype"], tf.int32, tf.int32, tf.int32, tf.float32],
                            stateful = False), num_parallel = 8)
                if self.params["max_duration"] > 0:
                    self._dataset = self._dataset.filter(
                            lambda x, x_len, y, y_len, duration:
                            tf.less_equal(duration, self.params["max_duration"]))
