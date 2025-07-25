# Tutorial 6: MISAR-seq (Mouse brain)

DePass was applied to the mouse brain dataset (2,129 spatial points) for cross-modal integration. This tutorial demonstrates the identification of spatial domains within brain tissues.

### Preparation


```python
import scanpy as sc
import torch
import random
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from DePass.utils import *
fix_seed(2024)  

# Environment configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.environ['R_HOME'] = '/home/jyx/anaconda3/envs/DePass/lib/R'

path = '../data/dataset_MISARseq_E18_5_S1'
os.mkdir(path) if not os.path.exists(path) else print(f"The directory '{path}' already exists.\n")
```

    The directory '/home/jyx/DePass-main/outputs/dataset_MISARseq_E18_5_S1/run' already exists.
    


### Loading and Preprocessing


```python
# read data
adata_omics1 = sc.read_h5ad('../data/dataset_MISARseq_E18_5_S1/adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad('../data/dataset_MISARseq_E18_5_S1/adata_ATAC.h5ad')

preprocess_data(adata=adata_omics1,modality='rna')
preprocess_data(adata=adata_omics2,modality='atac')

adata_omics1.obsm['input_feat'] = adata_omics1.obsm['X_norm'].copy()
adata_omics2.obsm['input_feat'] = adata_omics2.obsm['X_lsi'].copy()
data = {'rna': adata_omics1, 'atac': adata_omics2}
```

### Training the model


```python
from DePass.model import DePass
model = DePass(data, data_type='spatial',device=device)
output = model.train()
```

    [Config]
    Modalities: 2 | Data: spatial | Device: NVIDIA GeForce RTX 4090 
    
    [Initializing]
    Graph Construction : Running...
    Graph Construction : Done!
    Data Enhancement : Running...
    Data Enhancement : Done!
    
    [Training]
    Model training starts...


    100%|██████████| 200/200 [00:01<00:00, 103.99it/s]

    Model training finished!
    


    



```python
adata = adata_omics1.copy()
adata.obsm['DePass'] = model.embedding
adata.obsm['alpha'] = model.alpha 
```

### Detect spatial domain 


```python
from DePass.utils import *
clustering(adata=adata,n_clusters=14,key='DePass',add_key='DePass',method='mclust',use_pca=True)
```

    R[write to console]:                    __           __ 
       ____ ___  _____/ /_  _______/ /_
      / __ `__ \/ ___/ / / / / ___/ __/
     / / / / / / /__/ / /_/ (__  ) /_  
    /_/ /_/ /_/\___/_/\__,_/____/\__/   version 6.1.1
    Type 'citation("mclust")' for citing this R package in publications.
    


    fitting ...
      |======================================================================| 100%



```python
from DePass.utils import super_eval
import pandas as pd
re = super_eval(adata.obs['DePass'],adata.obs['Y'])
pd.DataFrame(list(re.items()), columns=['metric', 'Value']).to_csv(path + '/re.csv', sep='\t', index=True, float_format='%.6f')
print(re)

```

    {'AMI': 0.5947587217123534, 'NMI': 0.6020910379792522, 'ARI': 0.5695781369003142, 'Homogeneity': 0.6359861462799504, 'V-measure': 0.6020910379792523, 'Mutual Information': 1.3462211692280366}



```python
from DePass.analyze_utils import plot_spatial

plot_spatial(
    adata,
    color='DePass',
    save_path=path,
    save_name='DePass',
    title="DePass",
    s=35,
    dpi=300,
    format='pdf',
    show=True,
)

```


    
![png](Tutorial6_files/Tutorial6_12_0.png)
    



```python
plot_spatial(
    adata,
    color='Combined_Clusters_annotation',
    save_path=path,
    save_name='Combined_Clusters_annotation',
    title="celltype",
    s=35,
    dpi=300,
    format='pdf',
    show=True,
)

```


    
![png](Tutorial6_files/Tutorial6_13_0.png)
    



```python
adata.write_h5ad(path + '/adata.h5ad')  
```
