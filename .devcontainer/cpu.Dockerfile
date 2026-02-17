FROM python:3.11-slim

ENV ACCELERATOR=cpu
ARG DEBIAN_FRONTEND=noninteractive

# Force the system 'python3' command to point to the image's Python 3.11
RUN ln -sf /usr/local/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/local/bin/python3.11 /usr/bin/python

# Install system dependencies including compilation tools
RUN apt update && \
    apt install -y \
        build-essential \
        git \
        git-lfs \
        pkg-config \
        libfreetype6-dev \
        libpng-dev \
        python3-opencv && \
    rm -rf /var/lib/apt/lists/*

# Pin setuptools<82 (tensorboard needs pkg_resources), pip<24, wheel<0.45
RUN pip install --upgrade "pip<24" "wheel<0.45" "setuptools<82" tensorboard

# Install pytorch CPU
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install any system package dependencies
RUN apt update &&\
    apt install -y \
        python3-opencv