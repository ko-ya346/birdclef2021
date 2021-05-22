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
from tqdm.auto import tqdm

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
from src.model import TimmSED 
from src.dataset import TestDataset, get_transforms
from src.utils import timer

def prepare_model_for_inference(model, path: Path):
    if not torch.cuda.is_available():
        ckpt = torch.load(path, map_location="cpu")
    else:
        ckpt = torch.load(path)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def prediction_for_clip(test_df: pd.DataFrame,
                        clip: np.ndarray,
                        model,
                        threshold=0.5,
                        pred_keys='clipwise_output'):

    dataset = TestDataset(df=test_df,
                          clip=clip,
                          waveform_transforms=get_transforms(phase="test"))
    loader = torchdata.DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    prediction_dict = {}
    for image, row_id in tqdm(loader):
        row_id = row_id[0]
        image = image.to(device)

        with torch.no_grad():
            prediction = model(image)
            proba = prediction[pred_keys].detach().cpu().numpy().reshape(-1)

        events = proba >= threshold
        labels = np.argwhere(events).reshape(-1).tolist()

        if len(labels) == 0:
            prediction_dict[row_id] = "nocall"
        else:
            labels_str_list = list(map(lambda x: CFG.target_columns[x], labels))
            label_string = " ".join(labels_str_list)
            prediction_dict[row_id] = label_string
    return prediction_dict

def prediction(test_audios,
	       logger,
               weights_path: Path,
               threshold=0.5,
               pred_keys='clipwise_output'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimmSED(base_model_name=CFG.base_model_name,
                    pretrained=False,
                    num_classes=CFG.num_classes,
                    in_channels=CFG.in_channels)
    model = prepare_model_for_inference(model, weights_path).to(device)

    warnings.filterwarnings("ignore")
    prediction_dfs = []
    for audio_path in test_audios:
        with timer(f"Loading {str(audio_path)}", logger):
            clip, _ = sf.read(audio_path)

        seconds = []
        row_ids = []
        for second in range(5, 605, 5):
            row_id = "_".join(audio_path.name.split("_")[:2]) + f"_{second}"
            seconds.append(second)
            row_ids.append(row_id)

        test_df = pd.DataFrame({
            "row_id": row_ids,
            "seconds": seconds
        })
        with timer(f"Prediction on {audio_path}", logger):
            prediction_dict = prediction_for_clip(test_df,
                                                  clip=clip,
                                                  model=model,
                                                  threshold=threshold,
                                                  pred_keys=pred_keys)
        row_id = list(prediction_dict.keys())
        birds = list(prediction_dict.values())
        prediction_df = pd.DataFrame({
            "row_id": row_id,
            "birds": birds
        })
        prediction_dfs.append(prediction_df)

    prediction_df = pd.concat(prediction_dfs, axis=0, sort=False).reset_index(drop=True)
    return prediction_df
