
## DePass Installation

It is preferred to create a new environment for DePass.

```bash
# Create and activate a new conda environment
conda create -n DePass python==3.8.20
conda activate DePass
```

DePass is available on PyPI, and could be installed using:

```bash
pip install DePass
```

Installation via Github is also provided:

```bash
git clone https://github.com/yuanxiangjiang/DePass
cd DePass
pip install DePass-0.0.16-py3-none-any.whl
```

Additionally, because DePass leverages mclust for clustering, installing R, the rpy2 Python interface, and the mclust R package is recommended.
```bash
conda install -c conda-forge r-base rpy2 
conda install conda-forge::r-mclust
```

Install pytorch geometric and its CUDA 12.1 extensions

```bash
pip install torch==2.4.1+cu121
pip install torch-geometric==2.3.1
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```