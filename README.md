# stillleben

## Installation

1. Install Anaconda with Python 3.6 or higher: https://www.anaconda.com/distribution/#linux \
   I recommend *not* to initialize the Anaconda environment
2. Install CUDA + CUDNN (if you have an NVIDIA GPU)
3. Install PyTorch from source: https://github.com/pytorch/pytorch#from-source
4. Activate your Anaconda environment and install stillleben:
```bash
source ~/anaconda3/bin/activate
python setup.py install
```
5. Run the `display_mesh.py` example:
```bash
python tools/display_mesh.py tests/stanford_bunny/scene.gltf
```
