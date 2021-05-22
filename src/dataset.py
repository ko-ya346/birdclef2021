import gc
import os
import math
import random
import warnings
import librosa

import albumentations as A
import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
import colorednoise as cn

from pathlib import Path
from typing import List

from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from catalyst.core import Callback, CallbackOrder, IRunner
from catalyst.dl import Runner, SupervisedRunner
from sklearn import model_selection
from sklearn import metrics
from timm.models.layers import SelectAdaptivePool2d
from torch.optim.optimizer import Optimizer
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchlibrosa.augmentation import SpecAugmentation

from src.config import CFG


class WaveformDataset(torchdata.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: Path,
                 waveform_transforms=None,
                 period=20,
                 validation=False):
        self.df = df
        self.datadir = datadir
        self.waveform_transforms = waveform_transforms
        self.period = period
        self.validation = validation

    def _normalize(self, arr):
        max_arr = np.max(arr)
        min_arr = np.min(arr)
        return (arr - min_arr) / (max_arr - min_arr)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["filename"] + '.npy'
        ebird_code = sample["primary_label"]

        y = np.load(self.datadir / ebird_code / wav_name)

        # todo 引数で与えるように変更
        sr = 32000

#        if self.waveform_transforms:
#            y = self.waveform_transforms(y)

        len_y = y.shape[1]
        
        effective_length = sr * self.period // CFG.n_fft
        
        if len_y < effective_length:
            new_y = np.zeros((y.shape[0], effective_length), 
                    dtype=y.dtype)
            if not self.validation:
                start = np.random.randint(effective_length - len_y)
            else:
                start = 0
            new_y[:, start:start + len_y] = y
            y = new_y
        elif len_y > effective_length:
            if not self.validation:
                start = np.random.randint(len_y - effective_length)
            else:
                start = 0
            y = y[:, start:start + effective_length]

        y = y.astype(np.float32)
        y = np.nan_to_num(y)
        y = y.transpose(1, 0)
        y = y[np.newaxis, ...]

        labels = np.zeros(len(CFG.target_columns), dtype=float)
        labels[CFG.target_columns.index(ebird_code)] = 1.0

        return {
            "image": y,
            "targets": labels
        }


class TestDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, clip: np.ndarray,
                 waveform_transforms=None):
        self.df = df
        self.clip = clip
        self.waveform_transforms=waveform_transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        SR = 32000
        sample = self.df.loc[idx, :]
        row_id = sample.row_id

        end_seconds = int(sample.seconds)
        start_seconds = int(end_seconds - 5)

        start_index = SR * start_seconds
        end_index = SR * end_seconds

        y = self.clip[start_index:end_index].astype(np.float32)

        y = np.nan_to_num(y)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        y = np.nan_to_num(y)
        return y, row_id


def get_transforms(phase: str):
    transforms = CFG.transforms
    if transforms is None:
        return None
    else:
        if transforms[phase] is None:
            return None
        trns_list = []
        for trns_conf in transforms[phase]:
            trns_name = trns_conf["name"]
            trns_params = {} if trns_conf.get("params") is None else \
                trns_conf["params"]
            if globals().get(trns_name) is not None:
                trns_cls = globals()[trns_name]
                trns_list.append(trns_cls(**trns_params))

        if len(trns_list) > 0:
            return Compose(trns_list)
        else:
            return None

def get_waveform_transforms(config: dict, phase: str):
    return get_transforms(config, phase)

def get_spectrogram_transforms(config: dict, phase: str):
    transforms = config.get('transforms')
#    transforms = config.get('spectrogram_transforms')
    if transforms is None:
        return None
    else:
        if transforms[phase] is None:
            return None
        trns_list = []
        for trns_conf in transforms[phase]:
            trns_name = trns_conf["name"]
            trns_params = {} if trns_conf.get("params") is None else \
                trns_conf["params"]
            if hasattr(A, trns_name):
                trns_cls = A.__getattribute__(trns_name)
                trns_list.append(trns_cls(**trns_params))
            else:
                trns_cls = globals().get(trns_name)
                if trns_cls is not None:
                    trns_list.append(trns_cls(**trns_params))

        if len(trns_list) > 0:
            return A.Compose(trns_list, p=1.0)
        else:
            return None

class Normalize:
    def __call__(self, y: np.ndarray):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)


class NewNormalize:
    def __call__(self, y: np.ndarray):
        y_mm = y - y.mean()
        return y_mm / y_mm.abs().max()


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError



class MelSpectrogram(AudioTransform):
    '''
    メルスペクトログラムに変換する
    '''
    def __init__(self, always_apply=True, p=0.5, sr=32000,
            n_mels=128, fmin=20, fmax=16001):
        super().__init__(always_apply, p)
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    def apply(self, y: np.ndarray, **params):
        melspec = librosa.feature.melspectrogram(
                y, sr=self.sr, n_mels=self.n_mels,
                fmin=self.fmin, fmax=self.fmax, **params
                )
        augmented = librosa.power_to_db(melspec).astype(np.float32)        
        return augmented

class MFCC(AudioTransform):
    '''
    MFCCに変換する
    '''
    def __init__(self, always_apply=True, p=0.5, sr=32000,
            n_mels=128, fmin=20, fmax=16001, n_mfcc=20,
            dct_type=2, norm='ortho'):
        super().__init__(always_apply, p)
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm

    def apply(self, y: np.ndarray, **params):
        mfcc = librosa.feature.mfcc(
                y, sr=self.sr, n_mels=self.n_mels,
                fmin=self.fmin, fmax=self.fmax, 
                n_mfcc=self.n_mfcc, dct_type=self.dct_type,
                norm=self.norm, **params
                )
        augmented = mfcc.astype(np.float32)        
        return augmented 

class NoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=0.5, sr=32000):
        super().__init__(always_apply, p)

        self.noise_level = (0.01, max_noise_level)
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_level).astype(y.dtype)
        return augmented


class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20, sr=32000):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented


class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20, sr=32000):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented


class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_range=5, sr=32000):
        super().__init__(always_apply, p)
        self.max_range = max_range
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        n_steps = np.random.randint(-self.max_range, self.max_range)
        augmented = librosa.effects.pitch_shift(y, self.sr, n_steps)
        return augmented


class TimeStretch(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_rate=1, sr=32000):
        super().__init__(always_apply, p)
        self.max_rate = max_rate
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate)
        return augmented


def _db2float(db: float, amplitude=True):
    if amplitude:
        return 10**(db / 20)
    else:
        return 10 ** (db / 10)


def volume_down(y: np.ndarray, db: float):
    """
    Low level API for decreasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to decrease
    Returns
    -------
    applied: numpy.ndarray
        audio with decreased volume
    """
    applied = y * _db2float(-db)
    return applied


def volume_up(y: np.ndarray, db: float):
    """
    Low level API for increasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to increase
    Returns
    -------
    applied: numpy.ndarray
        audio with increased volume
    """
    applied = y * _db2float(db)
    return applied


class RandomVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        if db >= 0:
            return volume_up(y, db)
        else:
            return volume_down(y, db)


class OneOf:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        n_trns = len(self.transforms)
        trns_idx = np.random.choice(n_trns)
        trns = self.transforms[trns_idx]
        y = trns(y)
        return y


class CosineVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
        dbs = _db2float(cosine * db)
        return y * dbs


def drop_stripes(image: np.ndarray, dim: int, drop_width: int, stripes_num: int):
    total_width = image.shape[dim]
    lowest_value = image.min()
    for _ in range(stripes_num):
        distance = np.random.randint(low=0, high=drop_width, size=(1,))[0]
        begin = np.random.randint(
            low=0, high=total_width - distance, size=(1,))[0]

        if dim == 0:
            image[begin:begin + distance] = lowest_value
        elif dim == 1:
            image[:, begin + distance] = lowest_value
        elif dim == 2:
            image[:, :, begin + distance] = lowest_value
    return image


class TimeFreqMasking(ImageOnlyTransform):
    def __init__(self,
                 time_drop_width: int,
                 time_stripes_num: int,
                 freq_drop_width: int,
                 freq_stripes_num: int,
                 always_apply=False,
                 p=0.5):
        super().__init__(always_apply, p)
        self.time_drop_width = time_drop_width
        self.time_stripes_num = time_stripes_num
        self.freq_drop_width = freq_drop_width
        self.freq_stripes_num = freq_stripes_num

    def apply(self, img, **params):
        img_ = img.copy()
        if img.ndim == 2:
            img_ = drop_stripes(
                img_, dim=0, drop_width=self.freq_drop_width, 
                stripes_num=self.freq_stripes_num)
            img_ = drop_stripes(
                img_, dim=1, drop_width=self.time_drop_width, 
                stripes_num=self.time_stripes_num)
        return img_
 
