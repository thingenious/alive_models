# BASE_IMAGEs:
# depending on the cuda version (1x.y)
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags

# nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04
# nvcr.io/nvidia/cuda:12.3.2-devel-ubuntu22.04
# nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04
# nvcr.io/nvidia/cuda:12.1.1-devel-ubuntu22.04
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM --platform=linux/amd64 $BASE_IMAGE

ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV BASE_IMAGE=$BASE_IMAGE

RUN echo "Building image with base image: $BASE_IMAGE"

# environment variables
ENV TZ=UTC
# make sure python output is sent straight to terminal
ENV PYTHONUNBUFFERED 1
# make sure apt doesn't ask questions
ENV DEBIAN_FRONTEND="noninteractive"
ENV DEBCONF_NONINTERACTIVE_SEEN true

# install dependencies
RUN apt update && \
    apt install -y --no-install-recommends \
    libpython3.10 \
    python3-pip \
    python3-dev \
    curl \
    git \
    tzdata \
    locales \
    ca-certificates \
    ffmpeg \
    openssl \
    tini && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/archives/*

# install system-wide, else pytriton might complain
# (no module named numpy/zmq)
RUN pip install --upgrade pip wheel setuptools numpy zmq

RUN mkdir -p /opt/ml /root/.local/bin && \
    echo 'PATH=/root/.local/bin:$PATH' >> /root/.bashrc && \
    # remove setopt command from bashrc
    # to avoid error: setopt: command not found
    sed -i '/shopt/d' /root/.bashrc

ENV PATH=/root/.local/bin:$PATH

RUN git clone https://github.com/omry/omegaconf -b v2.0.6 && \
    sed -i 's/PyYAML>=5.1.*/PyYAML>=5.1/' omegaconf/requirements/base.txt && \
    pip install ./omegaconf && \
    rm -rf omegaconf
COPY requirements/main.txt /tmp/requirements.txt
RUN pip install --user -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

RUN pip uninstall -y opencv-python > /dev/null 2>&1 && \
    pip install --user opencv-python-headless

# install flash-attn
RUN MAX_JOBS=4 pip install flash-attn --no-build-isolation

RUN proto_version=$(python3 -c 'import importlib.metadata;print(importlib.metadata.version("protobuf"))') && \
    pip install git+https://github.com/huggingface/parler-tts.git && \
    pip install --upgrade protobuf==$proto_version

# ports
# the defauls that pytriton uses
ARG HTTP_PORT=8000
ENV HTTP_PORT=$HTTP_PORT
ARG GRPC_PORT=8001
ENV GRPC_PORT=$GRPC_PORT
ARG METRICS_PORT=8002
ENV METRICS_PORT=$METRICS_PORT
ARG SAGEMAKER_PORT=8080
ENV SAGEMAKER_PORT=$SAGEMAKER_PORT

EXPOSE $HTTP_PORT
EXPOSE $GRPC_PORT
EXPOSE $METRICS_PORT
EXPOSE $SAGEMAKER_PORT

# app
COPY app /opt/ml/app
WORKDIR /opt/ml

COPY scripts/start.sh /root/.local/bin/serve
COPY scripts/init.sh /opt/ml/init.sh
#
# For SageMaker
RUN chmod +x /root/.local/bin/serve /opt/ml/init.sh
RUN echo "#!/bin/env sh" > /root/.local/bin/train && \
    echo "echo 'train is not supported'" >> /root/.local/bin/train && \
    echo "exit 1" >> /root/.local/bin/train && \
    chmod +x /root/.local/bin/train

ENV TOKENIZERS_PARALLELISM=true
ENV TINI_SUBREAPER=true

# entrypoint
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["serve"]
