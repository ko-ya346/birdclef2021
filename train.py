import gc
import os
import math
import random
import warnings

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


from pathlib import Path

INPUT_DIR = './input'

import sys
sys.path.append('src')

from src.config import CFG
from src.dataset import *
from src.model import *
from src.lossess import *
from src.get_model import *
from src.utils import *

DEBUG = False
if DEBUG:
    CFG.epochs = 1

warnings.filterwarnings("ignore")

logdir = Path("output")
logdir.mkdir(exist_ok=True, parents=True)
if (logdir / "train.log").exists():
    os.remove(logdir / "train.log")
logger = init_logger(log_file=logdir / "train.log")

# environment
set_seed(CFG.seed)
device = get_device()

# validation
splitter = getattr(model_selection, CFG.split)(**CFG.split_params)

# data
train = pd.read_csv(CFG.train_csv)

# main loop
for i, (trn_idx, val_idx) in enumerate(splitter.split(train, y=train["primary_label"])):
    if i not in CFG.folds:
        continue
    logger.info("=" * 120)
    logger.info(f"Fold {i} Training")
    logger.info("=" * 120)

    trn_df = train.loc[trn_idx, :].reset_index(drop=True)
    val_df = train.loc[val_idx, :].reset_index(drop=True)

    loaders = {
        phase: torchdata.DataLoader(
            WaveformDataset(
                df_,
                CFG.train_datadir,
                img_size=CFG.img_size,
                waveform_transforms=get_transforms(phase),
                period=CFG.period,
                validation=(phase == "valid")
            ),
            **CFG.loader_params[phase])  # type: ignore
        for phase, df_ in zip(["train", "valid"], [trn_df, val_df])
    }

    model = TimmSED(
        base_model_name=CFG.base_model_name,
        pretrained=CFG.pretrained,
        num_classes=CFG.num_classes,
        in_channels=CFG.in_channels)
    criterion = get_criterion()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    callbacks = get_callbacks()
    runner = get_runner(device)
    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=CFG.epochs,
        verbose=True,
        logdir=logdir / f"fold{i}",
        callbacks=callbacks,
        main_metric=CFG.main_metric,
        minimize_metric=CFG.minimize_metric)

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
