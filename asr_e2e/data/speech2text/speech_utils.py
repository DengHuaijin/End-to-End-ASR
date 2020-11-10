from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import os
import math
import h5py

import numpy as np
import scipy.io.wavfile as wave
import resampy as rs
import librosa

WINDOW_FNS = {"hanning": np.hanning, "hamming": np.hamming, "none": None}

class PreprocessOnTheFlyException(Exception):
    pass

class RegenerateCacheException(Exception):
    pass

def load_features(path, data_format):

    if data_format == "hdf5":
        with h5py.File(path + ".hdf5", "r") as hf5_file:
            features = hf5_file["features"][:]
            duration = hf5_file["features"].attrs["duration"]
    elif data_format == "npy":
        features, duration = np.load(path + "npy")
    else:
        raise ValueError("Invalid data format for caching: {}\n options: hdf5, npy".format(data_format))

    return feature, duration

def save_features(features, duration, path, data_format, verbose = False):

    if verbose:
        print("Saving to: ", path)

    if data_format == "hdf5":
        with h5py.File(path + "hdf5", "w") as hf5_file:
            dset = hf5_file.create_dataset("features", data = features)
            dset.attrs["duration"] = duration
    elif data_format == "npy":
        np.save(path + ".npy", [features, duration])
    else:
        raise ValueError("Invalid data format for caching: {}\n options: hdf5, npy".format(data_format))

def get_preprocessed_data_path(filename, params):
    
    filename = os.path.realpath(filename)
    ignored_params = ["cache_features", "cache_format", "cache_regenerate",
                     "vocab_file", "dataset_files", "shuffle", "batch_size",
                     "max_duration", "mode", "interactive", "autoregressive",
                     "char2idx", "tgt_vocab_size", "idx2char", "dtype"]

    def fix_kv(text):
        text = str(text)
        text = text.replace("speed_perturbation_ratio", "sp") \
                   .replace("noise_level_min", "nlmin") \
                   .replace("noise_level_max", "nlmax") \
                   .replace("add_derivatives", "d") \
                   .replace("add_second_derivatives", "dd")
        return text

    preprocess_id = "-".join(
            [fix_kv(k) + "_" + fix_kv(v) for k,v in params.items() if k not in ignored_params])

    preprocessed_dir = os,path.dirname(filename).replace("wav", "preprocessed-" + preprocess_id)

    preprocessed_path = os.path.join(preprocessed_dir, os.path.basename(filename).replace("wav", ""))

    if not os.path.exists(preprocessed_dir):
        os,mkdirs(preprocessed_dir)

    return preprocessed_path


def get_speech_features(signal, sample_freq, params):
    
    backend = params.get("backend", "librosa")

    features_type = params.get("input_type", "spectrogram")
    num_features = params["num_audio_features"]
    window_size = params.get("window_size", 20e-3)
    window_stride = params.get("window_stride", 10e-3)
    augmentation = params.get("augmentation", None)

    if backend == "librosa":
        window_fn = WINDOW_FNS[params.get("window", "hanning")]
        dither = params.get("dither", 0)
        num_fft = params.get("num_fft", None)
        norm_per_feature = params.get("norm_per_feature", False)
        mel_basis = params.get("mel_basis", None)
        gain = params.get("gain")
        mean = params.get("features_mean")
        std_dev = params.get("features_std_dev")
        features, duration = get_speech_features_librosa(
                signal, sample_freq, num_features, features_type,
                window_size, window_stride, augmentation, window_fn = window_fn,
                dither = dither, norm_per_feature = norm_per_feature, 
                num_fft = num_fft, mel_basis = mel_basis, gain = gain,
                mean = mean, std_dev = std_dev)
    else:
        raise ValueError("librosa backend only")

    return features, duration

def get_speech_features_librosa(signal, sample_freq, num_features,
                                features_type = "spectrogram",
                                window_size = 20e-3,
                                window_stride = 10e-3,
                                augmentation = None,
                                window_fn = np.hanning,
                                num_fft = None,
                                dither = 0,
                                norm_per_feature = False,
                                mel_basis = None,
                                gain = None,
                                mean = None,
                                std_dev = None):
    
    signal = normalize_signal(signal.astype(np.float32), gain)

    if augmentation:
        signal = augment_audio_signal(signal, sample_freq, augmentation)
    
    audio_duration = len(signal) * 1 / sample_freq

    n_window_size = int(sample_freq * window_size)
    n_window_stride = int(sample_freq * window_stride)
    num_fft = num_fft or 2**math.ceil(math.log2(window_size * sample_freq))

    if dither > 0:
        signal += dither * np.random.randn(*signal.shape)

    if features_type == "spectrogram":
        powspec = np.square(np.abs(librosa.core.stft(
            signal, n_fft = n_window_size,
            hop_length = n_window_stride,
            win_length = n_window_size,
            center = True,
            window = window_fn)))
        powspec[powspec <= 1e-30] = 1e-30
        features = 10 * np.log10(powspec.T)

        assert num_features <= n_window_size // 2 + 1, "num_features for spectrogram should be <= n_window_size // 2 + 1"
        # cut high freq part
        features = features[:, :num_features]
    
    elif features_type == "mfcc":
        signal = preemphasis(signal, coeff = 0.97)
        S = np.square(
                np.abs(
                    librosa.core.stft(signal, n_fft = num_fft,
                                      hop_length = n_window_stride,
                                      win_length = n_window_size,
                                      center = True, window = window_fn)))
        
        features = librosa.feature.mfcc(sr = sample_freq, S = S, n_mfcc = num_features, n_mels = 2*num_features).T
    
    elif features_type == "logfbank":
        signal = preemphasis(signal, coeff = 0.97)

        S = np.square(
                np.abs(
                    librosa.core.stft(signal, n_fft = num_fft,
                                      hop_length = int(window_stride * sample_freq),
                                      win_length = int(window_size * sample_freq),
                                      center = True, window = window_fn)))
        if mel_basis is None:
            mel_basis = librosa.filters.mel(sample_freq, num_fft, n_mels = num_features, 
                                            fmin = 0, fmax = int(sample_freq/2))
        features = np.log(np.dot(mel_basis, S) + 1e-20).T

    else:
        raise ValueError("Unknown features type: {}".format(features_type))

    norm_axis = 0 if norm_per_feature else None
    if mean is None:
        mean = np.mean(features, axis = norm_axis)
    if std_dev is None:
        std_dev = np.std(features, axis = norm_axis)
    
    features = (features - mean) / std_dev

    if augmentation:
        n_freq_mask = augmentation.get("n_freq_mask", 0)
        n_time_mask = augmentation.egt("n_time_mask", 0)
        width_freq_mask = augmentation.get("width_freq_mask", 10)
        width_time_mask = augmentation.get("width_time_mask", 50)

        for idx in range(n_freq_mask):
            freq_band = np.random.randint(width_freq_mask + 1)
            freq_base = np.random.randint(0, features.shape[1] - freq_band)
            features[:, :freq_base:freq_base+freq_band] = 0
        
        for idx in range(n_time_mask):
            time_band = np.random.randint(width_time_mask + 1)
            if features.shape[0] - time_band > 0:
                time_base = np.random.randint(features.shape[0] - time_band)
                features[time_base:time_base+time_band, :] = 0
    
    return features, audio_duration


def preemphasis(signal, coeff = 0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def normalize_signal(signal, gain = None):
    """
    Normalize to [-1, 1]
    """
    if gain is None:
        gain = 1.0 / (np.max(np.abs(signal)) + 1e-5)
    return signal * gain

def augment_audio_signal(signal_float, sample_freq, augmentation):

    if "speed_perturbation_ratio" in augmentation:
        stretch_amount = -1
        if isinstance(augmentation["speed_perturbation_ratio"], list):
            stretch_amount = np.random.choice(augmentation["speed_perturbation_ratio"])
        elif augmentation["speed_perturbation_ratio"] > 0:
            """
            np,random.rand() -> U[0,1]
            U[1-speed_perturbation_ratio, 1+speed_perturbation_ratio]
            """
            stretch_amount = 1.0 + (2.0 * np.random.rand() - 1.0) * augmentation["speed_perturbation_ratio"]
        if stretch_amount > 0:
            signal_float = rs.resample(signal_float,
                                       sample_freq,
                                       int(sample_freq * stretch_amount),
                                       filter = "kaiser_best")
    if "noise_level_min" in augmentation and "noise_level_max" in augmentation:
        noise_level_db = np.random.randint(low = augmentation["noise_level_min"], 
                                           high = augmentation["noise_level_max"])
        signal_float += np.random.randint(signal_float.shape[0]) * 10.0 ** (noise_level_db / 20)
    
    return signal_float

def get_speech_features_from_file(filename, params):

    cache_features = params.get("cache_features", False)
    cache_format = params.get("cache_format", "hdf5")
    cache_regenerate = params.get("cache_regenerate", False)
        
    try:
        if not cache_features:
            raise PreprocessOnTheFlyException(
                    "on-the-fly preprocessing enforced with cache_features == True")

        if cache_regenerate:
            raise RegenerateCacheException("regenerating cache...")

        preprocessed_data_path = get_preprocessed_data_path(filename, params)
        features, duration = load_features(preprocessed_data_path, data_format = cache_format)
    
    except PreprocessOnTheFlyException:

        sample_freq, signal = wave.read(filename)

        if sample_freq != params["sample_freq"]:
            raise ValueError(
                    "The sampling frequency set in params {} does not match"
                    " the frequency {} read from file {}".format(params["sample_freq"], sample_freq, filename))

        features, duration = get_speech_features(signal, sample_freq, params)

    except (OSError, FileNotFoundError, RegenerateCacheException):
        sample_freq, signal = wave.read(filename)

        if sample_freq != params["sample_freq"]:
            raise ValueError(
                    "The sampling frequency set in params {} does not match"
                    " the frequency {} read from file {}".format(params["sample_freq"], sample_freq, filename))

        features, duration = get_speech_features(signal, sample_freq, params)
        preprocessed_data_path = get_preprocessed_data_path(filename, params)
        save_features(features, duration, preprocessed_data_path, data_format = cache_format)


    return features, duration


