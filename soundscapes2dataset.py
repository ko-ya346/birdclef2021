import os
from pathlib import Path
from glob import glob

import librosa
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm
import pandas as pd

from src.config import CFG


INPUT_DIR  = 'input/'
OUTPUT_DIR = 'output/train_short_audio/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

train_labels = pd.read_csv(INPUT_DIR + 'train_soundscape_labels.csv')
train_meta = pd.read_csv(INPUT_DIR + 'train_metadata.csv')

##### dataset作成時に読み込むdataframeを作る #####
train_meta['wave_path'] = train_meta['primary_label'] + '/' + train_meta['filename']

# train_soundscapesのlabelがlistで入ってるのでlistに変換する
train_meta['primary_label'] = [[val] for val in train_meta.primary_label.values]

train_labels['primary_label'] = train_labels.birds.str.split(' ')
train_labels['wave_path'] = (train_labels['audio_id'].astype(str) 
                            + '_' + train_labels['site'] 
                            + '/' + train_labels['seconds'].astype(str) + '.ogg')

cols = ['primary_label', 'wave_path']
label_df = pd.concat(
        [train_meta.loc[:, cols], 
         train_labels.loc[:, cols]]
        )
print(label_df.shape)

label_df.to_csv(f'output/label.csv', index=False)

SR = 32000

##### train_soundscapesにあるファイルを5秒刻みにして保存 #####
for idx in tqdm(range(len(train_labels))):
    sample = train_labels.loc[idx, :]
    row_id = sample.row_id
    audio_id = sample.audio_id
    site = sample.site
    seconds = sample.seconds

    save_dir = f'output/train_short_audio/{audio_id}_{site}'
    # ファイル単位でディレクトリを作る
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ogg_path = list(glob(f'input/train_soundscapes/{audio_id}_{site}*.ogg'))[0]
    data, _ = sf.read(ogg_path)

    for end_seconds in range(5, 605, 5):
        start_seconds = end_seconds - 5
        start_index = SR * start_seconds
        end_index = SR * end_seconds
        
        audio = data[start_index: end_index]
        mel = librosa.feature.melspectrogram(y=audio, 
            sr=SR,
            n_fft=CFG.n_fft,
            hop_length=CFG.hop_length,
            )
        logmel = librosa.power_to_db(mel, ref=np.max)

        np.save(f'{save_dir}/{seconds}.ogg.npy', logmel.astype(np.uint8))

