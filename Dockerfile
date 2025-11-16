# syntax=docker/dockerfile:1
FROM python:3.9-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libegl1 \
        libgl1-mesa-dev \
        libglew-dev \
        libosmesa6-dev \
        patchelf \
        libxrandr-dev \
        libxrender-dev \
        pkg-config \
        swig \
        unzip \
        wget \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/lib

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade "pip<24.1"

# Gym 0.21.0 ships only an sdist whose setup.py declares an invalid
# opencv-python constraint. Patch it once here so future installs succeed.
RUN pip download gym==0.21.0 -d /tmp \
    && tar -xzf /tmp/gym-0.21.0.tar.gz -C /tmp \
    && sed -i 's/opencv-python>=3\./opencv-python>=3.0.0/' /tmp/gym-0.21.0/setup.py \
    && pip install /tmp/gym-0.21.0 \
    && rm -rf /tmp/gym-0.21.0 /tmp/gym-0.21.0.tar.gz

# Install CPU-only PyTorch wheels to avoid downloading CUDA toolkits.
RUN pip install --no-cache-dir \
        torch==2.1.2 \
        torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /workspace

ENV PYTHONPATH=/workspace

CMD ["/bin/bash"]
