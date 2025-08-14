
# DePass

DePass is a dual-enhanced graph learning framework designed for integrated analysis of both single-cell and spatial paired multi-omics data. It flexibly supports diverse modality combinations.

<p align="left">
  <img src=image.png width="700"/>
</p>


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
git clone https://github.com/zhanglabNKU/DePass.git
cd DePass
pip install depass-0.0.20-py3-none-any.whl
```

Additionally, because DePass leverages mclust for clustering, installing R, the rpy2 Python interface, and the mclust R package is recommended.
```bash
conda install -c conda-forge r-base rpy2 
conda install conda-forge::r-mclust
```

Install pytorch, pytorch geometric

```bash
pip install torch==2.4.1
pip install torch-geometric==2.3.1
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

---

## Requirements

```text
anndata==0.11.4
hnswlib==0.8.0
matplotlib==3.7.1
numpy==1.23.5
pandas==1.5.3
scanpy==1.11.3
scikit_learn==1.3.0
scipy==1.16.0
seaborn==0.13.2
setuptools==75.3.2
tqdm==4.65.0
torch==2.4.1
torch_geometric==2.3.1
torch_scatter==2.1.2+pt24cu121
torch-sparse==0.6.18+pt24cu121
torch-cluster==1.6.3+pt24cu121
torch-spline-conv==1.2.2+pt24cu121
R==4.3.1
rpy2==3.5.11
jupyter
mclust
```

---


## Data

The raw data used in this study is freely available at [Data](https://drive.google.com/drive/folders/1NfBDU-1E2T7CerViyK2TvbnWfsta6Q3r?usp=drive_link).
