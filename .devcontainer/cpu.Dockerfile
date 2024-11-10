FROM python

ENV ACCELERATOR=cpu
ARG DEBIAN_FRONTEND=noninteractive

# Install pytorch CPU
RUN pip3 install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install any system package dependencies
RUN apt update &&\
    apt install -y \
        python3-opencv