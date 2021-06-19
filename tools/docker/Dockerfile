FROM xqms/pytorch:v1.8.1-devel
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         cmake ninja-build libglvnd-dev libegl1-mesa-dev libgl1-mesa-dev libassimp-dev libbullet-dev \
         texlive texlive-latex-extra texlive-fonts-extra libgs9 \
     && rm -rf /var/lib/apt/lists/*

RUN  /opt/conda/bin/conda install -y cmake python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     /opt/conda/bin/conda install -y sphinx sphinx_rtd_theme && \
     /opt/conda/bin/conda install -y opencv && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends tex-gyre \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN chmod -R a+w .

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# For some reason, the nvidia docker runtime does not mount/copy this file.
# Without it, GLVND does not find the nvidia EGL drivers.
COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Small utility to confirm EGL works
COPY egl_info.cpp /tmp/egl_info.cpp
RUN g++ -std=c++17 -Wall -o /usr/bin/egl_info /tmp/egl_info.cpp -lX11 -lEGL


