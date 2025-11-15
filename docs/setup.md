create conda environment
```bash
git clone --recurse-submodules https://github.com/arthurfenderbucker/open_perception.git
conda create -n open_perception python=3.11.7
conda activate open_perception
```
<!-- install torch>=2.5.1 with cuda 12.4
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
conda install nvidia/label/cuda-12.4.0::cuda-nvcc # install the proper nvcc, needs to be the same as torch
conda install nvidia/label/cuda-12.4.0::cuda-toolkit
conda install nvidia/label/cuda-12.4.0::cuda-runtime
```

setup the environment variables for the cuda libraries
```bash
export CUDA_HOME=$CONDA_PREFIX
mkdir -p $CONDA_PREFIX/etc/conda/activate.d/
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d/
echo "export OLD_LDLIBRARY_PATH=\$LD_LIBRARY_PATH; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:\$LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export LD_LIBRARY_PATH=\$OLD_LDLIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
conda deactivate
conda activate open_perception
```

install the rest of the dependencies
```bash
pip install -r requirements.txt
```

download the [pre-trained models](../checkpoints/README.md)


```bash
export CUDA_HOME=$CONDA_PREFIX
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;8.9" CXX=/usr/bin/g++-11 LD=/usr/bin/g++-11 CC=/usr/bin/gcc-11 pip install git+https://github.com/IDEA-Research/GroundingDINO.git@d13643262e654c6e14b90995210da65d01f5783b#egg=groundingdino --no-cache-dir
``` -->
