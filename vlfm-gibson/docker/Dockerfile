# CUDA 11.8 base image
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Dependencies: Merged system installs to reduce layers
RUN apt-get update && apt-get install -y \
    wget git bzip2 ca-certificates \
    libx11-6 libxext6 libglib2.0-0 libegl1 libglvnd0 libgl1 libglx-mesa0 libgles2-mesa-dev \
    # X11 Forwarding / rendering fixes
    libsm6 libice6 libxkbcommon-x11-0 libxcb-xinerama0 libx11-xcb1 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-render-util0 libxcb-shape0 libxcb-shm0 libxcb-sync1 \
    libxcb-xfixes0 libxcb-dri3-0 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

#RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
#    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 

# # Create environment (Removed "conda tos" unless you are sure you need it)
# RUN conda config --add channels conda-forge && \
#     conda config --add channels defaults && \
#     conda create -n DCON python=3.9 cmake=3.14.0 -y && \
#     echo "conda activate DCON" >> ~/.bashrc

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda config --add channels conda-forge && \
    conda config --add channels defaults && \
    conda create -n DCON python=3.9 cmake=3.14.0 -y && \
    echo "conda activate DCON" >> ~/.bashrc

SHELL ["bash", "-c"]
ENV CONDA_DEFAULT_ENV=DCON
ENV PATH=/opt/conda/envs/DCON/bin:$PATH

# Install Conda basics
RUN source activate DCON && conda install -y \
    ipykernel ipython jupyter_client jupyter_core matplotlib-inline \
    && conda clean -afy

# 1. Install Torch 11.8 (Do this FIRST so it establishes the CUDA version)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. Install requirements (With numpy<2.0 pinned)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 3. tiny-cuda-nn
ENV TCNN_CUDA_ARCHITECTURES="89"
RUN pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# 4. Habitat-Sim (headless)
RUN git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-sim.git /workspace/habitat-sim && \
    cd /workspace/habitat-sim && \
    pip install -r requirements.txt && \
    python setup.py install --headless --bullet

# 5. Habitat-Lab
RUN git clone --branch stable https://github.com/facebookresearch/habitat-lab.git /workspace/habitat-lab && \
    pip install -e /workspace/habitat-lab/habitat-lab && \
    pip install -e /workspace/habitat-lab/habitat-baselines

# 6. Nerfstudio & Fixes
RUN pip install nerfstudio && \
    pip uninstall -y opencv-python-headless opencv-python && \
    pip install opencv-python nerfview transformers ftfy regex

RUN pip install "numpy<2.0"

#7. Segment Anything    
RUN pip install git+https://github.com/facebookresearch/segment-anything.git

WORKDIR /workspace

# EGL / GL environment
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV EGL_PLATFORM=surfaceless

# --- VLFM SYSTEM DEPENDENCIES ---
RUN apt-get update && apt-get install --no-install-recommends -qy \
    curl \
    apt-transport-https \
    gnupg \
    ca-certificates \
    git \
    tmux \
    software-properties-common \
    build-essential \
    libgl1-mesa-dev \
    libjpeg-dev \
    libglm-dev \
    libgl1-mesa-glx \
    libegl1-mesa-dev \
    mesa-utils \
    xorg-dev \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean all

# Adapt A: Explicitly use the Conda environment's pip binary
RUN /opt/conda/envs/DCON/bin/pip install \
    flask>=2.3.2 \
    seaborn>=0.12.2 \
    open3d>=0.17.0 \
    transformers==4.26.0 \
    six>=1.16.0 \
    contexttimer decord einops>=0.4.1 fairscale==0.4.4 ftfy iopath \
    omegaconf opencv-python-headless==4.5.5.64 opendatasets packaging \
    plotly pycocoevalcap pycocotools python-magic scikit-image \
    sentencepiece streamlit timm==0.4.12 webdataset wheel

RUN /opt/conda/envs/DCON/bin/pip install --no-deps salesforce-lavis==1.0.2

# Adapt A: Explicitly use the Conda environment's pip binary for Git repos
RUN /opt/conda/envs/DCON/bin/pip install \
                git+https://github.com/naokiyokoyama/frontier_exploration.git \
                git+https://github.com/ChaoningZhang/MobileSAM.git \
                git+https://github.com/naokiyokoyama/depth_camera_filtering \
                git+https://github.com/naokiyokoyama/bd_spot_wrapper.git

RUN /opt/conda/envs/DCON/bin/pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git

CMD ["/bin/bash"]