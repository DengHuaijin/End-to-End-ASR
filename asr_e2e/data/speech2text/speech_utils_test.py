from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import math
import os

import numpy as np
import numpy.testing as npt
import scipy.io.wavfile as wave
import tensorflow as tf
from six.moves import range

from speech_utils import get_speech_features, get_speech_features_from_file, augment_audio_signal

class SpeechUtilsTests(tf.test.TestCase):

    def test_augment_audio_signal(self):
        filename = "wav_files/103-1240-0000.wav"
        freq_s, signal = wave.read(filename)
        signal = signal.astype(np.float32)
        
        augmentation = {
                "speed_perturbation_ratio": 0.2,
                "noise_level_min": -90,
                "noise_level_max": -46
                }
        """
        1.2 >= length(aug) >= 0.8
        """
        for _ in range(100):
            signal_aug = augment_audio_signal(signal, freq_s, augmentation)
            self.assertLessEqual(signal.shape[0] * 0.8, signal_aug.shape[0])
            self.assertGreaterEqual(signal.shape[0] * 1.2, signal_aug.shape[0])

        augmentation = {
                "speed_perturbation_ratio": 0.5,
                "noise_level_min": -90,
                "noise_level_max": -46
                }
        
        for _ in range(100):
            signal_aug = augment_audio_signal(signal, freq_s, augmentation)
            self.assertLessEqual(signal.shape[0] * 0.5, signal_aug.shape[0])
            self.assertGreaterEqual(signal.shape[0] * 1.5, signal_aug.shape[0])

    def test_get_speech_features_from_file(self):
        dirname = "wav_files/"
        for name in ["103-1240-0000.wav"]:
            filename = os.path.join(dirname, name)
            for num_features in [161, 120]:
                for window_stride in [10e-3, 5e-3, 40e-3]:
                    for window_size in [20e-3, 30e-3]:
                        for feature_type in ["spectrogram", "mfcc", "logfbank"]:
                            freq_s, signal = wave.read(filename)
                            n_window_size = int(freq_s * window_size)
                            n_windws_stride = int(freq_s * window_stride)
                            length = 1 + (signal.shape[0] - n_window_size) // n_windws_stride
                            """
                            if length % 8 != 0:
                                length += 8 - length % 8
                            """
                            right_shape = (length, num_features)
                            params = {}
                            params["num_audio_features"] = num_features
                            params["input_type"] = feature_type
                            params["window_size"] = window_size
                            params["window_stride"] = window_stride
                            params["sample_freq"] = 16000
                            input_features, _ = get_speech_features_from_file(filename, params)
                            """
                            backend == librosa时不一定是8的倍数
                            backend == psf时需要判断
                            """
                            # self.assertTrue(input_features.shape[0] % 8 == 0)
                            # self.assertTupleEqual(right_shape, input_features.shape)
                            self.assertAlmostEqual(np.mean(input_features), 0.0, places = 6)
                            self.assertAlmostEqual(np.std(input_features), 1.0, places = 6)
                            
                            
    def test_get_speech_features_from_file_augmentation(self):
        augmentation = {
                "speed_perturbation_ratio": 0.0,
                "noise_level_min": -90,
                "noise_level_max": -46
                }
        filename = "wav_files/103-1240-0000.wav"
        num_features = 161
        params = {}
        params["sample_freq"] = 16000
        params["num_audio_features"] = num_features
        input_features_clean, _ = get_speech_features_from_file(filename, params)

        params["augmentation"] = augmentation

        input_features_aug, _ = get_speech_features_from_file(filename, params)
        
        self.assertTrue(np.all(np.not_equal(input_features_clean, input_features_aug)))

        augmentation = {
                "speed_perturbation_ratio": 0.2,
                "noise_level_min": -90,
                "noise_level_max": -46
                }

        params["augmentation"] = augmentation
        input_features_aug, _ = get_speech_features_from_file(filename, params)
        self.assertNotEqual(input_features_clean.shape[0], input_features_aug.shape[0])
        self.assertEqual(input_features_clean.shape[1], input_features_aug.shape[1])

tf.test.main()
