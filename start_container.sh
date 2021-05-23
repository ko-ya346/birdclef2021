docker run -it --gpus all --shm-size=10gb --mount type=bind,source=/home/kouya-takahashi/kaggle/birdclef2021,destination=/workspace --name gpu-birdclef kaggle2:birdclef /bin/bash
