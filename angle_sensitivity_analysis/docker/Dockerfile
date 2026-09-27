# Rendering-only environment for the viewpoint sensitivity dataset.
# Deliberately isolated from the VLFM production image — this container
# only ever needs habitat-sim + basic image I/O, nothing model-related.

FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget bzip2 git build-essential cmake \
    libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa \
    && rm -rf /var/lib/apt/lists/*

# Miniconda (habitat-sim is distributed via conda-forge / aihabitat channel)
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

RUN conda create -n habitat_render python=3.9 -y
SHELL ["conda", "run", "-n", "habitat_render", "/bin/bash", "-c"]

# Pin habitat-sim to match whatever version your VLFM eval harness already
# validated against (0.2.4 per your existing setup) so object/scene loading
# behaves identically to what you've already debugged.
RUN conda install -y habitat-sim=0.2.4 withbullet -c conda-forge -c aihabitat
RUN pip install numpy-quaternion pillow pandas

WORKDIR /workspace
COPY render_multiview_dataset.py /workspace/render_multiview_dataset.py
COPY run_all_scenes.sh /workspace/run_all_scenes.sh

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "habitat_render", "python", "render_multiview_dataset.py"]