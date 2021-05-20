import os
from pathlib import Path
from glob import glob

import librosa
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm
# from torchlibrosa.stft import LogmelFilterBank, Spectrogram

from src.config import CFG

OUTPUT_DIR = 'output/train_short_audio/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

AUDIO_PATHS = list(glob('input/train_short_audio/*/*.ogg'))

for i in tqdm(range(len(AUDIO_PATHS))):
    filepath = AUDIO_PATHS[i]
    dir_name, filename = filepath.split('/')[-2:]

    save_path = Path(OUTPUT_DIR, dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    audio, sr = sf.read(filepath)

    mel = librosa.feature.melspectrogram(y=audio, 
            sr=sr,
            n_fft=CFG.n_fft,
            hop_length=CFG.hop_length,
            )
    logmel = np.log(mel)

    np.save(Path(OUTPUT_DIR, dir_name, filename+'.npy'), logmel.astype(np.uint8))
