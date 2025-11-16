# syntax=docker/dockerfile:1
FROM python:3.9-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_VERSION=2.1.0 \
    MUJOCO_INSTALL_DIR=/root/.mujoco \
    MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210 \
    MUJOCO_PY_MJKEY_PATH=/root/.mujoco/mjkey.txt \
    MUJOCO_GL=osmesa \
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
        libxrandr-dev \
        libxrender-dev \
        pkg-config \
        swig \
        unzip \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p ${MUJOCO_INSTALL_DIR} \
    && wget -qO /tmp/mujoco.tar.gz https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco210-linux-x86_64.tar.gz \
    && tar -C ${MUJOCO_INSTALL_DIR} -xzf /tmp/mujoco.tar.gz \
    && rm /tmp/mujoco.tar.gz \
    && ln -s ${MUJOCO_INSTALL_DIR}/mujoco210 ${MUJOCO_INSTALL_DIR}/mujoco \
    && touch ${MUJOCO_PY_MJKEY_PATH}

ENV LD_LIBRARY_PATH=${MUJOCO_PY_MUJOCO_PATH}/bin:/usr/local/lib

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
        torch==2.1.2+cpu \
        torchvision==0.16.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /workspace

ENV PYTHONPATH=/workspace

CMD ["/bin/bash"]
