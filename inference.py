import os
import sys
from pathlib import Path
import pandas as pd

sys.path.append('src')
from src.utils import get_logger, set_seed
from src.get_model import *
from src.model import *
from src.dataset import *
from src.config import *

from src.utils import *
from src.lossess import *


INPUT_DIR = './input'
logger = get_logger("main.log")
set_seed(1213)

TARGET_SR = 32000
TEST = (len(list(Path(f"{INPUT_DIR}/test_soundscapes/").glob("*.ogg"))) != 0)
if TEST:
    DATADIR = Path(f"{INPUT_DIR}/test_soundscapes/")
else:
    DATADIR = Path(f"{INPUT_DIR}/train_soundscapes/")

all_audios = list(DATADIR.glob("*.ogg"))
all_audio_ids = ["_".join(audio_id.name.split("_")[:2]) for audio_id in all_audios]
submission_df = pd.DataFrame({
    "row_id": all_audio_ids
})
print(submission_df.head())

weights_path = Path(f"{INPUT_DIR}/birdclef2021-effnetb0-starter-weight/best.pth")
submission = prediction(test_audios=all_audios,
                        weights_path=weights_path,
                        threshold=0.5)
submission.to_csv("submission.csv", index=False)
