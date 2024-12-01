FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y libglib2.0-0 && apt-get install ffmpeg libsm6 libxext6  -y
