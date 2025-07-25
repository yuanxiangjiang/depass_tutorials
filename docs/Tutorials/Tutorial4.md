# Tutorial 4: SAM (PD Mouse B)

DePass was applied to the mouse brain dataset (2,820 spatial spots) for cross-modal integration and data enhancement. This tutorial demonstrates: 
1. Spatial domain identification in mouse brain tissue using DePass.
2. Validation of data enhancement through comparative analysis of biomarker log2 fold-changes (LogFC) and spatial expression patterns.

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
  
path = '../data/dataset_PD_Mouse_Brain_B'
os.mkdir(path) if not os.path.exists(path) else print(f"The directory '{path}' already exists.\n")
```

    The directory '/home/jyx/DePass-main/outputs/dataset_PD_Mouse_Brain_B/run' already exists.
    


### Loading and Preprocessing


```python
adata_omics1 = sc.read_h5ad('../data/dataset_PD_Mouse_Brain_B/adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad('../data/dataset_PD_Mouse_Brain_B/adata_meta.h5ad')

preprocess_data(adata=adata_omics1,modality='rna')
preprocess_data(adata=adata_omics2,modality='metabolite')

adata_omics1.obsm['input_feat'] = adata_omics1.obsm['X_norm'].copy()
adata_omics2.obsm['input_feat'] = adata_omics2.obsm['X_norm'].copy()
data = {'rna': adata_omics1, 'metabolite': adata_omics2}
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


    100%|██████████| 200/200 [00:02<00:00, 89.18it/s] 

    Model training finished!
    


    



```python
adata = adata_omics1.copy()
adata.obsm['DePass'] = model.embedding
adata.obsm['alpha'] = model.alpha 
```

### Detect spatial domain 


```python
from DePass.utils import *
clustering(adata=adata,n_clusters=11,key='DePass',add_key='DePass',method='mclust',use_pca=True)
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
print(re)
df = pd.DataFrame(list(re.items()), columns=['metric', 'Value']).to_csv(path + '/re.csv', sep='\t', index=True, float_format='%.6f')
```

    {'AMI': 0.6614983541972036, 'NMI': 0.6646111905762093, 'ARI': 0.5002914670734389, 'Homogeneity': 0.7321120361539022, 'V-measure': 0.6646111905762092, 'Mutual Information': 1.339706403559318}



```python
from DePass.analyze_utils import plot_spatial

plot_spatial(
    adata,
    color='DePass',
    save_path=path,
    save_name='DePass',
    title="DePass",
    s=25,
    dpi=300,
    format='pdf',
    show=True,
)

```


    
![png](Tutorial4_files/Tutorial4_12_0.png)
    



```python
plot_spatial(
    adata,
    color='annotations',
    save_path=path,
    save_name='annotations',
    title="annotations",
    s=25,
    dpi=300,
    format='pdf',
    show=True,
)

```


    
![png](Tutorial4_files/Tutorial4_13_0.png)
    



```python
adata.write(path+'/adata.h5ad')
```

### dpt_pseudotime


```python
import anndata as ad
adata_dpt = ad.AnnData(X=adata.obsm['DePass'],  obsm={'spatial': adata.obsm['spatial'].copy()})
```


```python
adata_dpt.write(path+'/adata_dpt')
```


```python
from DePass.analyze_utils import plot_spatial
sc.pp.neighbors(adata_dpt,n_neighbors=30)
sc.tl.diffmap(adata_dpt)
adata_dpt.uns['iroot'] = 0
sc.tl.dpt(adata_dpt, n_branchings=0, n_dcs=10)
plot_spatial(adata_dpt,color='dpt_pseudotime',save_path=path,save_name='psM_DePass',title="DePass",s=25,dpi=300,format='pdf',color_map='plasma',show=True)

```


    
![png](Tutorial4_files/Tutorial4_18_0.png)
    

