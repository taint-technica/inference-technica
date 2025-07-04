FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /opt/inference

# Copy only requirements files first for better Docker layer caching
COPY xinference/deploy/docker/requirements-base.txt xinference/deploy/docker/requirements-base.txt
COPY xinference/deploy/docker/requirements-ml.txt xinference/deploy/docker/requirements-ml.txt
COPY xinference/deploy/docker/requirements-models.txt xinference/deploy/docker/requirements-models.txt

# RUN apt-get update && apt-get -y install wget
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
# RUN dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb
# RUN apt-get update && apt-get -y install cuda-toolkit-12-4

# ENV CUDA_HOME /usr/local/cuda
# ENV PATH ${CUDA_HOME}/bin:${PATH}
# ENV LD_LIBRARY_PATH /usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}

ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION 14.21.1

RUN apt-get -y update \
  && apt install -y wget curl procps git libgl1 libnuma1 \
  # upgrade libstdc++ and libc for llama-cpp-python
  && printf "\ndeb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse" >> /etc/apt/sources.list \
  && apt-get -y update \
  && apt-get install -y --only-upgrade libstdc++6 && apt install -y libc6 \
  && mkdir -p $NVM_DIR \
  && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash \
  && . $NVM_DIR/nvm.sh \
  && nvm install $NODE_VERSION \
  && nvm alias default $NODE_VERSION \
  && nvm use default \
  && apt-get -yq clean

ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD TRUE
ENV UV_HTTP_TIMEOUT=300

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG PIP_INDEX=https://pypi.org/simple

# Install python 3.10
RUN apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && apt install python3.10 python3.10-venv python3.10-dev python3-pip -y

RUN pip install --upgrade -i "$PIP_INDEX" pip setuptools wheel && \
    uv pip install --system -i "$PIP_INDEX" "diskcache>=5.6.1" "jinja2>=2.11.3" && \
    uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install --system flash-attn==2.7.4.post1 --no-build-isolation && \

    # Uninstall torch
    uv pip uninstall --system torch torchvision torchaudio && \
    uv pip install --system -i "$PIP_INDEX" --upgrade -r /opt/inference/xinference/deploy/docker/requirements-base.txt && \
    uv pip install --system -i "$PIP_INDEX" --upgrade -r /opt/inference/xinference/deploy/docker/requirements-ml.txt && \
    uv pip install --system -i "$PIP_INDEX" --upgrade -r /opt/inference/xinference/deploy/docker/requirements-models.txt && \
    uv pip install --system -i "$PIP_INDEX" transformers>=4.51.3 && \

    # Install the forked sglang version from taint-technica
    cd /opt && \
    git clone https://github.com/taint-technica/sglang-technica.git && \
    cd sglang-technica && \
    uv pip install --system -e "python[all]" && \
    cd /opt/inference

    # # Install vllm
    # uv pip install --system vllm --torch-backend=auto

# Copy the rest of the source code (after heavy installations for better caching)
COPY . /opt/inference

# Build the web UI and install the package
RUN python3 setup.py build_web && \
    (git restore . || true) && \
    uv pip install --system -i "$PIP_INDEX" --no-deps "." && \
    uv pip uninstall --system xllamacpp && \
    uv pip install --system "xllamacpp>=0.1.21" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu124 && \
    uv clean

# Install Miniforge3 (only for FFmpeg, do not replace system Python)
RUN apt-get install -y wget && \
    wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3.sh -b -p /opt/conda && \
    rm Miniforge3.sh

RUN /opt/conda/bin/conda create -n ffmpeg-env -c conda-forge 'ffmpeg<7' -y && \
    ln -s /opt/conda/envs/ffmpeg-env/bin/ffmpeg /usr/local/bin/ffmpeg && \
    ln -s /opt/conda/envs/ffmpeg-env/bin/ffprobe /usr/local/bin/ffprobe && \
    /opt/conda/bin/conda clean --all -y

# Overwrite the entrypoint of vllm's base image
ENTRYPOINT ["xinference-local"]
CMD ["--host", "0.0.0.0", "--port", "9997"]
