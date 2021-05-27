import gc
import os
import warnings

import albumentations as A
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
                        models: list,
                        threshold=0.5,
                        batch_size=1,
                        pred_keys='clipwise_output'):

    dataset = TestDataset(df=test_df,
                          clip=clip,
                          waveform_transforms=get_transforms(phase="test"))
    loader = torchdata.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prediction_dict = {}
    for image, row_id in tqdm(loader):
        row_id = row_id[0]
        image = image.to(device)

        proba = np.zeros(397)
        for model in models:
            with torch.no_grad():
                prediction = model(image)
                proba += prediction[pred_keys].detach().cpu().numpy().reshape(-1)

            del model, prediction; gc.collect()
        del image; gc.collect()

        proba /= len(models)
        events = proba >= threshold
        labels = np.argwhere(events).reshape(-1).tolist()

        if len(labels) == 0:
            prediction_dict[row_id] = "nocall"
        else:
            labels_str_list = list(map(lambda x: CFG.target_columns[x], labels))
            label_string = " ".join(labels_str_list)
            prediction_dict[row_id] = label_string

    del loader; gc.collect()

    return prediction_dict

def prediction(test_audios,
	       logger,
               weights_paths: list,
               threshold=0.5,
               pred_keys='clipwise_output',
               TEST=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimmSED(base_model_name=CFG.base_model_name,
                    pretrained=False,
                    num_classes=CFG.num_classes,
                    in_channels=CFG.in_channels,
                    TEST=TEST)

    models = []
    for weights_path in weights_paths:
        load_model = prepare_model_for_inference(model, weights_path).to(device)
        models.append(load_model)
        del load_model; gc.collect()

    warnings.filterwarnings("ignore")
#    prediction_dfs = []
    prediction_row_id = []
    prediction_birds = []

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
                                                  models=models,
                                                  threshold=threshold,
                                                  pred_keys=pred_keys)
        row_id = prediction_dict.keys()
        birds = prediction_dict.values()

        prediction_row_id.append(row_id)
        prediction_birds.append(birds)

        del clip; gc.collect()

    prediction_df = pd.DataFrame(
            {"row_id": prediction_row_id,
             "birds":  orediction_birds}
            )
    return prediction_df
