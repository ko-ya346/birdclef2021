docker run -it --gpus all --mount type=bind,source=/home/kouya-takahashi/kaggle/birdclef2021,destination=/workspace --name tensorboard -p 8889:8889 tensorflow/tensorflow:latest-gpu-py3 /bin/bash
