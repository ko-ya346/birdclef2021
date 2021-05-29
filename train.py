import gc
import os
import warnings

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata

from pathlib import Path
from typing import List

from sklearn import model_selection


from src.config import CFG
from src.dataset import *
from src.model import *
from src.lossess import *
from src.get_model import *
from src.utils import *


DEBUG = CFG.debug


def main():
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
    print(train.shape)

    # main loop
    for i, (trn_idx, val_idx) in enumerate(splitter.split(train, y=train["primary_label"])):
        if i not in CFG.folds:
            continue
        logger.info("=" * 120)
        logger.info(f"Fold {i} Training")
        logger.info("=" * 120)

        trn_df = train.loc[trn_idx, :].reset_index(drop=True)
        val_df = train.loc[val_idx, :].reset_index(drop=True)
        print(trn_df.shape, val_df.shape)

        loaders = {
            phase: torchdata.DataLoader(
                WaveformDataset(
                    df_,
                    CFG.train_datadir,
                    waveform_transforms=get_transforms(phase),
                    period=CFG.period,
                    validation=(phase == "valid")
                ),
                **CFG.loader_params[phase])  # type: ignore
            for phase, df_ in zip(["train", "valid"], [trn_df, val_df])
        }
        
        # dataloaderのサイズを確認
        tmp = loaders['train'].__iter__()
        print(len(tmp))

        x, y = tmp.next().values()
        print(x.shape)
        print(y.shape)
#        exit()

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

    text = 'finish train'
    send_line_message(text)

if __name__ == '__main__':
    main()
